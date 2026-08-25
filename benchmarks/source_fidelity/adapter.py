from framework.mechanism import MechanismProbeAdapter


class SourceFidelityAdapter(MechanismProbeAdapter):
    def __init__(self, env_file: str) -> None:
        super().__init__(env_file=env_file, benchmark_name="source_fidelity")
