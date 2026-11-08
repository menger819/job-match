from jobmatch.collectors.base import BlockedError, Collector, PoliteClient
from jobmatch.collectors.cake import CakeCollector
from jobmatch.collectors.jobs_104 import Jobs104Collector
from jobmatch.collectors.linkedin import LinkedInCollector

REGISTRY = {
    "104": Jobs104Collector,
    "cake": CakeCollector,
    "linkedin": LinkedInCollector,
}

__all__ = ["REGISTRY", "BlockedError", "Collector", "PoliteClient"]
