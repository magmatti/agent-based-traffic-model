import logging


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger("traffic_sim")
