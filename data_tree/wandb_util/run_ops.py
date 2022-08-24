from dataclasses import dataclass

from wandb.apis.public import Run


@dataclass
class WandbRunOps:
    src: Run

    def as_dict(self):
        return dict(
            id=self.src.id,
            name=self.src.name,
            path=self.src.path,
            state=self.src.state
        )