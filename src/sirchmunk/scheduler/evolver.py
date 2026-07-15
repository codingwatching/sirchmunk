# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Knowledge cluster evolver — organize knowledge clusters at runtime.

Periodically connects, merges, refreshes, and organises KnowledgeClusters
that are created at search time, so the knowledge graph improves with use.

The evolver is invoked via ``KnowledgeEvolver.step(cluster)`` after each
``AgenticSearch.search()`` call. It checks four trigger conditions and
executes the corresponding phase:

* Phase 1: connect & merge (every step, when new cluster count >= _EVO_CONNECT_MERGE_COUNT)
* Phase 2: edge refresh (every step, when refresh count >= _EVO_EDGE_REFRESH_COUNT)
* Phase 3: meta cluster detection (>= _EVO_STEP_INTERVAL steps since last run,
and >= _EVO_META_DETECTION_COUNT change in non-meta cluster count)
* Phase 4: global update (>= _EVO_STEP_INTERVAL steps since last run, 
and >= _EVO_GLOBAL_UPDATE_COUNT change in non-meta cluster count)
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

import igraph as ig
import leidenalg

from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.schema.knowledge import KnowledgeCluster, Lifecycle, WeakSemanticEdge
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.utils.embedding_util import EmbeddingUtil, compute_text_hash
from sirchmunk.utils import LogCallback, create_logger

# Minimum number of new clusters required to trigger connect and merge phase
_EVO_CONNECT_MERGE_COUNT = 5

# Similarity threshold for connecting clusters through weak semantic edges
_EVO_CONNECT_SIMILARITY_THRESHOLD = 0.60

# Similarity threshold for merging clusters
_EVO_MERGE_SIMILARITY_THRESHOLD = 0.90

# Minimum number of edge refreshes required to trigger edge refresh phase
_EVO_EDGE_REFRESH_COUNT = 20

# Maximum number of clusters to consider for edge refresh in a single step
_EVO_EDGE_REFRESH_TOPK = 10

# Minimum number of clusters required to trigger meta cluster detection phase
_EVO_META_DETECTION_COUNT = 100

# Minimum number of clusters required to trigger global update phase
_EVO_GLOBAL_UPDATE_COUNT = 500

# Step interval for meta cluster detection and global update phases
_EVO_STEP_INTERVAL = 10


@dataclass
class EvolveManifest:
    """Tracks the state of the knowledge evolver for resume and incremental processing."""

    version: str = "1.0"
    current_step: int = 0
    cluster_ids_buffer: List[str] = field(default_factory=list)
    last_evolve_at: Optional[str] = None

    connect_merge_cluster_ids: List[str] = field(default_factory=list)
    edge_refresh_cluster_ids: List[str] = field(default_factory=list)

    last_meta_detection_step: int = 0
    last_meta_detection_count: int = 0
    last_global_update_step: int = 0
    last_global_update_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EvolveManifest":
        data = json.loads(json_str)
        return cls(
            version=data.get("version", "1.0"),
            current_step=data.get("current_step", 0),
            cluster_ids_buffer=data.get("cluster_ids_buffer", []),
            last_evolve_at=data.get("last_evolve_at"),
            connect_merge_cluster_ids=data.get("connect_merge_cluster_ids", []),
            edge_refresh_cluster_ids=data.get("edge_refresh_cluster_ids", []),
            last_meta_detection_step=data.get("last_meta_detection_step", 0),
            last_meta_detection_count=data.get("last_meta_detection_count", 0),
            last_global_update_step=data.get("last_global_update_step", 0),
            last_global_update_count=data.get("last_global_update_count", 0)
        )


class KnowledgeEvolver:
    """Periodically connects, merges, refreshes, and organises KnowledgeClusters at runtime."""

    def __init__(
        self,
        llm: OpenAIChat,
        embedding: Optional[EmbeddingUtil],
        knowledge_storage: KnowledgeStorage,
        work_path: Union[str, Path],
        log_callback: LogCallback = None,
    ):
        if embedding is None:
            raise ValueError("EmbeddingUtil instance is required for KnowledgeEvolver.")

        self._llm = llm
        self._embedding = embedding
        self._storage = knowledge_storage
        self._work_path = Path(work_path).expanduser().resolve()
        self._log = create_logger(log_callback=log_callback)

        self._evolve_dir = self._work_path / ".cache" / "evolve"
        self._evolve_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._evolve_dir / "manifest.json"
        self._manifest = self._load_manifest()

        # A lock to ensure that only one evolve step is processed at a time
        self._evolve_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    async def step(self, cluster: KnowledgeCluster):
        """
        Step the evolver with locking to ensure only one step is processed at a time.

        Args:
            cluster (KnowledgeCluster): The new knowledge cluster to process.
        """
        self._manifest.cluster_ids_buffer.append(cluster.id)
        self._save_manifest(self._manifest)

        if self._evolve_lock.locked():
            await self._log.info(
                "[Evolve] Evolve step is already in progress. "
                "The new cluster has been added to the buffer for later processing."
            )
            return

        async with self._evolve_lock:
            await self._log.info(
                "[Evolve] Starting evolver step with "
                f"{len(self._manifest.cluster_ids_buffer)} clusters in the buffer."
            )
            await self._step()
            await self._log.info("[Evolve] Evolver step completed.")

    async def _step(self):
        """Step the evolver to process the clusters in the buffer."""
        # Check the cluster is created or reused
        # created: | last_modified - created | < 1 second
        for cluster_id in self._manifest.cluster_ids_buffer:
            cluster = await self._storage.get(cluster_id)
            if cluster is None:
                await self._log.warning(
                    f"[Evolve] Cluster {cluster_id} not found in storage. Skipping."
                )
                continue

            cluster_last_modified = datetime.fromisoformat(cluster.last_modified)
            cluster_create = datetime.fromisoformat(cluster.create_time)
            if abs((cluster_last_modified - cluster_create).total_seconds()) < 1:
                # This cluster is newly created
                self._manifest.connect_merge_cluster_ids.append(cluster.id)
            else:
                # This cluster is reused with embedding refresh
                self._manifest.edge_refresh_cluster_ids.append(cluster.id)
        
        # Clear the buffer after processing
        self._manifest.cluster_ids_buffer.clear()

        # Update the current step count
        self._manifest.current_step += 1

        # Check and execute phases based on triggers
        has_evolved = False

        try:
            # Phase 1: Connect and Merge
            if self._should_connect_merge():
                await self._log.info(
                    f"[Evolve] Phase 1: Connect and Merge triggered "
                    f"at step {self._manifest.current_step}."
                )
                await self._connect_and_merge()
                has_evolved = True

            # Phase 2: Refresh Edges
            if self._should_refresh_edges():
                await self._log.info(
                    f"[Evolve] Phase 2: Refresh Edges triggered "
                    f"at step {self._manifest.current_step}."
                )
                await self._refresh_edges()
                has_evolved = True

            # Phase 3: Detect Meta Clusters
            if self._should_detect_meta_clusters():
                await self._log.info(
                    f"[Evolve] Phase 3: Detect Meta Clusters triggered "
                    f"at step {self._manifest.current_step}."
                )
                await self._detect_meta_clusters()
                has_evolved = True

            # Phase 4: Global Update
            if self._should_global_update():
                await self._log.info(
                    f"[Evolve] Phase 4: Global Update triggered "
                    f"at step {self._manifest.current_step}."
                )
                await self._global_update()
                has_evolved = True

        except Exception as e:
            await self._log.error(f"[Evolve] Error during evolve step: {e}")
        finally:
            # Update the last evolve timestamp and save the manifest
            if has_evolved:
                self._manifest.last_evolve_at = datetime.now(timezone.utc).isoformat()
            self._save_manifest(self._manifest)

    # ------------------------------------------------------------------ #
    #  Trigger Functions                                                 #
    # ------------------------------------------------------------------ #

    def _should_connect_merge(self) -> bool:
        """Check if the connect and merge phase should be triggered.
        
        Phase 1 (Connect and Merge) is triggered if the number of new clusters
        is greater than or equal to _EVO_CONNECT_MERGE_COUNT.
        """
        return len(self._manifest.connect_merge_cluster_ids) >= _EVO_CONNECT_MERGE_COUNT

    def _should_refresh_edges(self) -> bool:
        """Check if the edge refresh phase should be triggered.
        
        Phase 2 (Refresh Edges) is triggered if the number of clusters with
        refreshed embeddings is greater than or equal to _EVO_EDGE_REFRESH_COUNT.
        """
        return len(self._manifest.edge_refresh_cluster_ids) >= _EVO_EDGE_REFRESH_COUNT

    def _should_detect_meta_clusters(self) -> bool:
        """Check if the meta cluster detection phase should be triggered.
        
        Phase 3 (Detect Meta Clusters) is triggered if the number of steps
        since the last meta cluster detection is greater than or equal to
        _EVO_STEP_INTERVAL and the change in non-meta cluster count is greater
        than or equal to _EVO_META_DETECTION_COUNT.
        """
        # Check the step interval condition
        step_interval = self._manifest.current_step - self._manifest.last_meta_detection_step
        if step_interval < _EVO_STEP_INTERVAL:
            return False

        # Check the cluster count condition
        non_meta_cluster_count = self._get_non_meta_cluster_count()
        cluster_count_diff = abs(non_meta_cluster_count - self._manifest.last_meta_detection_count)
        return cluster_count_diff >= _EVO_META_DETECTION_COUNT

    def _should_global_update(self) -> bool:
        """Check if the global update phase should be triggered.
        
        Phase 4 (Global Update) is triggered if the number of steps since the
        last global update is greater than or equal to _EVO_STEP_INTERVAL and
        the change in non-meta cluster count is greater than or equal to
        _EVO_GLOBAL_UPDATE_COUNT.
        """
        # Check the step interval condition
        step_interval = self._manifest.current_step - self._manifest.last_global_update_step
        if step_interval < _EVO_STEP_INTERVAL:
            return False

        # Check the cluster count condition
        non_meta_cluster_count = self._get_non_meta_cluster_count()
        cluster_count_diff = abs(non_meta_cluster_count - self._manifest.last_global_update_count)
        return cluster_count_diff >= _EVO_GLOBAL_UPDATE_COUNT

    # ------------------------------------------------------------------ #
    #  Phase 1: Connect and Merge                                        #
    # ------------------------------------------------------------------ #

    async def _connect_and_merge(self):
        """Phase 1: Connect and Merge for newly created clusters.

        For each newly created cluster, find other clusters with similar embeddings.
        If the similarity exceeds _EVO_CONNECT_SIMILARITY_THRESHOLD, create a weak
        semantic edge between them. If the similarity exceeds _EVO_MERGE_SIMILARITY_THRESHOLD,
        merge the clusters into a single cluster. Update the manifest with the new
        cluster IDs and clear the list of newly created clusters.
        """
        # Since these clusters are already in the storage, we need to pick them out, 
        # and perform connect & merge during putting them back in
        for cluster_id in self._manifest.connect_merge_cluster_ids:
            cluster = self._storage.get(cluster_id)
            cluster_embedding = self._get_cluster_embedding(cluster_id)
            if cluster is None or cluster_embedding is None:
                continue

            # Find similar clusters based on embedding similarity
            # To avoid best match being in the list of newly created clusters, 
            # we search for top_k = len(connect_merge_cluster_ids) + 3
            best_match = None
            similar = self._storage.search_similar_clusters(
                query_embedding=cluster_embedding,
                top_k=len(self._manifest.connect_merge_cluster_ids) + 3,
                similarity_threshold=_EVO_CONNECT_SIMILARITY_THRESHOLD
            )
            # Similar is sorted by similarity, so the first one is the best match
            if similar:
                filtered_similar = [
                    s for s in similar
                    if s["id"] not in self._manifest.connect_merge_cluster_ids
                ]
                if filtered_similar:
                    best_match = filtered_similar[0]
            
            # This cluster is not similar enough to connect or merge
            if best_match is None:
                continue

            # If a best match is found, discuss whether to connect or merge
            if best_match["similarity"] >= _EVO_MERGE_SIMILARITY_THRESHOLD:
                # Merge clusters
                best_cluster = self._storage.get(best_match["id"])
                if best_cluster:
                    # Merge and update embedding
                    await self._merge_clusters(source=cluster, target=best_cluster)
            elif best_match["similarity"] >= _EVO_CONNECT_SIMILARITY_THRESHOLD:
                # Connect clusters with a weak semantic edge
                best_cluster = self._storage.get(best_match["id"])
                if best_cluster:
                    # Create a weak semantic edge between the two clusters
                    await self._create_weak_semantic_edge(
                        source=cluster,
                        target=best_cluster,
                        edge_source="embed_sim",
                        weight=best_match["similarity"]
                    )
            else:
                # No action needed if similarity is below the threshold
                continue

        # Clear the list of newly created clusters after processing
        self._manifest.connect_merge_cluster_ids.clear()

    # ------------------------------------------------------------------ #
    #  Phase 2: Refresh Edges                                            #
    # ------------------------------------------------------------------ #

    async def _refresh_edges(self):
        """Phase 2: Refresh Edges for clusters with updated embeddings.

        For each cluster with a refreshed embedding, find other clusters with similar embeddings.
        If the similarity exceeds _EVO_CONNECT_SIMILARITY_THRESHOLD, 
        create a weak semantic edge between them.
        """
        for cluster_id in self._manifest.edge_refresh_cluster_ids:
            cluster = self._storage.get(cluster_id)
            cluster_embedding = self._get_cluster_embedding(cluster_id)
            if cluster is None or cluster_embedding is None:
                continue

            # Find similar clusters based on embedding similarity
            similar = self._storage.search_similar_clusters(
                query_embedding=cluster_embedding,
                top_k=_EVO_EDGE_REFRESH_TOPK,
                similarity_threshold=_EVO_CONNECT_SIMILARITY_THRESHOLD
            )
            for similar_cluster in similar:
                if similar_cluster["id"] == cluster_id:
                    continue  # Skip self
                target_cluster = self._storage.get(similar_cluster["id"])
                if target_cluster:
                    await self._create_weak_semantic_edge(
                        source=cluster,
                        target=target_cluster,
                        edge_source="embed_sim",
                        weight=similar_cluster["similarity"]
                    )

    # ------------------------------------------------------------------ #
    #  Phase 3: Detect Meta Clusters                                     #
    # ------------------------------------------------------------------ #

    async def _detect_meta_clusters(self):
        """Phase 3: Detect Meta Clusters using Leiden community detection algorithm.

        Through the Leiden algorithm, the knowledge graph is partitioned into communities.
        For each community, if the number of clusters exceeds a threshold, a meta cluster
        is created to represent the community to reduce complexity during search.       
        """
        # Before running the meta cluster detection, we need to clean up existing meta clusters
        await self._clean_existing_meta_clusters()

        # Build edges for leiden algorithm
        edges: List[Tuple[str, str, float]] = []



    # ------------------------------------------------------------------ #
    #  Phase 4: Global Update                                            #
    # ------------------------------------------------------------------ #

    async def _global_update(self):
        pass

    # ------------------------------------------------------------------ #
    #  Shared Helper Functions                                           #
    # ------------------------------------------------------------------ #

    def _get_non_meta_cluster_count(self) -> int:
        """Get the count of non-meta clusters in the knowledge storage."""
        result = self._storage.db.fetch_one(
            f"SELECT COUNT(*) FROM {self._storage.table_name} "
            f"WHERE lifecycle != ?", [Lifecycle.META.name]
        )
        return result[0] if result else 0

    def _get_cluster_embedding(self, cluster_id: str) -> Optional[List[float]]:
        """Get the embedding vector for a given cluster ID."""
        try:
            row = self._storage.db.fetch_one(
                f"SELECT embedding_vector FROM {self._storage.table_name} "
                f"WHERE id = ?", [cluster_id]
            )
            return row[0] if row else None
        except Exception as e:
            self._log.error(f"Failed to fetch embedding for cluster {cluster_id}: {e}")
        return None

    # ------------------------------------------------------------------ #
    #  Phase 1 Helper Functions                                          #
    # ------------------------------------------------------------------ #

    async def _merge_clusters(self, source: KnowledgeCluster, target: KnowledgeCluster):
        """Merge source cluster into target cluster and update the storage."""
        merged_cluster = await self._storage.merge([target, source])
        # Update embedding with merged queries
        await self._update_cluster_embedding(merged_cluster)
        self._manifest.edge_refresh_cluster_ids.append(merged_cluster.id)

        # Update lifecycle if merge count exceeds threshold
        if merged_cluster.merge_count >= 3 and merged_cluster.lifecycle == Lifecycle.EMERGING:
            merged_cluster.lifecycle = Lifecycle.STABLE
        await self._storage.update(merged_cluster)

    async def _update_cluster_embedding(self, cluster: KnowledgeCluster):
        """Update the embedding of a cluster based on its queries."""
        if not cluster.queries:
            return
        try:
            combined_text = self._storage.combine_cluster_fields(cluster.queries)
            text_hash = compute_text_hash(combined_text)
            embedding_vector = (await self._embedding.embed([combined_text]))[0]

            await self._storage.store_embedding(
                cluster_id=cluster.id,
                embedding_vector=embedding_vector,
                embedding_model=self._embedding.model_id,
                embedding_text_hash=text_hash
            )
        except Exception as e:
            self._log.error(f"Failed to update embedding for cluster {cluster.id}: {e}")
            return

    # ------------------------------------------------------------------ #
    #  Phase 2 Helper Functions                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _add_edge(
        cluster: KnowledgeCluster,
        target_cluster_id: str, 
        edge_source: str,
        weight: float,
        use_max_weight: bool = True
    ):
        """Add a weak semantic edge to the cluster or update the weight"""
        for edge in cluster.related_clusters:
            if edge.target_cluster_id == target_cluster_id and edge.source == edge_source:
                if use_max_weight:
                    edge.weight = max(edge.weight, weight)
                else:
                    edge.weight = weight
                return
        cluster.related_clusters.append(
            WeakSemanticEdge(target_cluster_id=target_cluster_id, weight=weight, source=edge_source)
        )

    async def _create_weak_semantic_edge(
        self,
        source: KnowledgeCluster,
        target: KnowledgeCluster,
        edge_source: str,
        weight: float
    ):
        """Create a weak semantic edge between two clusters and update the storage."""
        self._add_edge(source, target.id, edge_source, weight)
        self._add_edge(target, source.id, edge_source, weight)
        # Update the clusters in storage
        await self._storage.update(source)
        await self._storage.update(target)

    # ------------------------------------------------------------------ #
    #  Phase 3 Helper Functions                                          #
    # ------------------------------------------------------------------ #

    async def _clean_existing_meta_clusters(self):
        """Clean up existing meta clusters from the knowledge storage."""
        meta_cluster_ids = self._storage.db.fetch_all(
            f"SELECT id FROM {self._storage.table_name} "
            f"WHERE lifecycle = ?", [Lifecycle.META.name]
        )
        for meta_cluster_id in meta_cluster_ids:
            await self._storage.remove(meta_cluster_id[0])

    # ------------------------------------------------------------------ #
    #  Manifest I/O                                                      #
    # ------------------------------------------------------------------ #
    def _load_manifest(self) -> EvolveManifest:
        if self._manifest_path.exists():
            try:
                return EvolveManifest.from_json(
                    self._manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass
        return EvolveManifest()

    def _save_manifest(self, manifest: EvolveManifest) -> None:
        """Atomically persist the manifest via write-to-tmp + rename.

        This prevents partial JSON on disk if the process is killed mid-write.
        """
        tmp_path = self._manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(manifest.to_json(), encoding="utf-8")
        tmp_path.replace(self._manifest_path)
