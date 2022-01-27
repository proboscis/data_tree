from dataclasses import dataclass

import wandb
from loguru import logger

from data_tree.wandb_util.artifact_getter import WandbArtifactGetter, PublicWandbArtifactGetter
from data_tree.wandb_util.artifact_identifier import ArtifactIdentifier, ArtifactMetadata


@dataclass
class RemoteArtifact:
    identifier: str
    type: str
    wandb_artifact_getter: WandbArtifactGetter
    wandb_download_dir: str

    # two ways to get remote artifact.
    # one is to traverse histories. => list of artifacts
    # one is to use with identifier. => strings
    # the former has metadata available
    # the latter has no metadata available unless you use it.
    # so while this class is meant to be used without "use"ing the artifact,
    # you need another class that implements different functionality.
    # there is no way to look at the artifact's metadata without using it.
    def download(self, path=None) -> str:
        path = path or f"{self.wandb_download_dir}/{self.identifier}"
        logger.info(f"downloading artifact:{self.identifier}")
        self._art = self.wandb_artifact_getter.use_artifact(
            self.identifier,
            type=self.type
        )
        return self._art.download(path)

    def get_path(self, name):
        return self._art.get_path(name)

    def get(self, object_name):
        """
        returns a python object associated with object_name
        :param object_name:
        :return:
        """
        return self._art.get(object_name)


@dataclass
class PublicRemoteArtifact:
    identifier: ArtifactIdentifier
    public_wandb_artifact_getter: PublicWandbArtifactGetter

    def download(self, path=None) -> str:
        art = self.public_wandb_artifact_getter.get_artifact(
            user="proboscis",
            project="archpainter",
            name=self.identifier.name,
            alias=self.identifier.version
        )
        return art.download(path=path)


@dataclass
class PublicRemoteArtifactFactory:
    public_wandb_artifact_getter: PublicWandbArtifactGetter

    def get(self, idt: ArtifactIdentifier):
        return PublicRemoteArtifact(idt, self.public_wandb_artifact_getter)


@dataclass
class RemoteArtifactFactory:
    wandb_artifact_getter: WandbArtifactGetter
    wandb_download_dir: str
    wandb_user: str
    wandb: wandb

    def from_identifier(self, identifier, type) -> RemoteArtifact:
        identifier = identifier.replace(f"{self.wandb_user}/{self.wandb.run.project}/", "")
        return RemoteArtifact(
            identifier,
            type, self.wandb_artifact_getter, wandb_download_dir=self.wandb_download_dir)

    def from_single_identifier(self, single_identifier: str) -> RemoteArtifact:
        """
        :param single_identifier: type:name:version
        :return:
        """
        _type, name, version = single_identifier.split(":")
        return self.from_identifier(name + ":" + version, type=_type)

    def from_artifact(self, art: "RemoteArtifactMetadata") -> RemoteArtifact:
        return RemoteArtifact(
            identifier=art.art.name,
            type=art.art.type,
            wandb_artifact_getter=self.wandb_artifact_getter,
            wandb_download_dir=self.wandb_download_dir
        )


@dataclass
class RemoteArtifactMetadata:
    art: wandb.Artifact
    remote_artifact_factory: RemoteArtifactFactory

    def to_remote_artifact(self) -> RemoteArtifact:
        return self.remote_artifact_factory.from_artifact(self)

    def to_artifact_metadata(self) -> ArtifactMetadata:
        return ArtifactMetadata(
            identifier=self.to_artifact_identifier(),
            description=self.art.description,
            metadata=self.metadata,
            aliases=[self.version]
        )

    def to_artifact_identifier(self) -> ArtifactIdentifier:
        # name already contains version so..
        name = self.name.replace(f":{self.version}","")
        return ArtifactIdentifier(name=name, type=self.type, version=self.version)

    @property
    def name(self):
        return self.art.name

    @property
    def metadata(self):
        return self.art.metadata

    @property
    def type(self):
        return self.art.type

    @property
    def version(self):
        return self.art.version


@dataclass
class RemoteArtifactMetadataFactory:
    remote_artifact_factory: RemoteArtifactFactory
    wandb_artifact_getter: WandbArtifactGetter

    def get(self, artifact: wandb.Artifact):
        return RemoteArtifactMetadata(artifact, self.remote_artifact_factory)

    def get2(self, idt: ArtifactIdentifier):
        art = self.wandb_artifact_getter.use_artifact2(idt)
        return self.get(art)
