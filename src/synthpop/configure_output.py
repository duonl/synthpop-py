import logging


def disable_logging_config() -> None:
    pass


class ConsoleConfiguration:

    def __init__(self, columns: list[str] | None = None,
                 include_general: bool = True,
                 level: int = logging.WARNING,
                 # TODO: is this right?
                 format: str = 'col: %(column)s -  %(message)s',
                 show_progressbar: bool = True
                 ):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, tb):
        pass

    @classmethod
    def configure_console_output(cls, columns: list[str] | None = None,
                                 include_general: bool = True,
                                 level: int = logging.WARNING,
                                 # TODO: is this right?
                                 format: str = 'col: %(column)s -  %(message)s',
                                 show_progressbar: bool = True,) -> None:
        pass


class FileOutputConfiguration:

    def __init__(self,
                 columns: list[str] | None = None,
                 include_general: bool = True,
                 level: int = logging.WARNING,
                 format: str = 'col: %(column)s -  %(message)s'):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, tb):
        pass
