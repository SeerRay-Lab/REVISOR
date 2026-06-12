from verl.workers.agent.tool_envs import ToolBase
import logging


logger = logging.getLogger(__name__)


class DummyTool(ToolBase):
    name = "dummy_tool"

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )

    def execute(self, *args, **kwargs) -> tuple:
        return "", 0.0, True, {}


    def reset(self, *args, **kwargs):
        return


if __name__ == "__main__":
    # Example usage (for testing)
    tool = DummyTool("dummy_tool", "A tool returns nothing", {})
