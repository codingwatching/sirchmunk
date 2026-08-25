from framework.mechanism import MechanismProbeAdapter


class SetupCostAdapter(MechanismProbeAdapter):
    def __init__(self, env_file: str) -> None:
        super().__init__(env_file=env_file, benchmark_name="setup_cost")
