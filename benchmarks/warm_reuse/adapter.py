from framework.mechanism import MechanismProbeAdapter


class WarmReuseAdapter(MechanismProbeAdapter):
    def __init__(self, env_file: str) -> None:
        super().__init__(env_file=env_file, benchmark_name="warm_reuse")
