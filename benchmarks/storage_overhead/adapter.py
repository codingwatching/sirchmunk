from framework.mechanism import MechanismProbeAdapter


class StorageOverheadAdapter(MechanismProbeAdapter):
    def __init__(self, env_file: str) -> None:
        super().__init__(env_file=env_file, benchmark_name="storage_overhead")
