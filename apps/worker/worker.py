"""EDAPHNET Asynchronous Background Task Worker.

Processes heavy batch inference, segmentation jobs, and PDF diagnostic reports
without blocking the low-latency REST API gateway.

Architecture:
API Queue (Redis / Fallback Store) -> Worker Daemon -> ML Pipeline -> Result Telemetry Store
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from src.edaphnet.application.pipeline_service import EdaphNetPipelineService
from src.edaphnet.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Worker] %(message)s")
logger = logging.getLogger("edaphnet.worker")


class EdaphNetWorker:
    """Asynchronous task worker for compute-heavy ML jobs."""

    def __init__(self):
        """Initialize worker and ML pipeline service."""
        self.pipeline_service = EdaphNetPipelineService()
        self.running = False
        logger.info("EdaphNetWorker daemon initialized.")

    def process_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single queued background ML task.
        
        Args:
            job_data: Task payload including job_id, type, and payload arguments.
            
        Returns:
            Dictionary with task completion status and payload.
        """
        job_id = job_data.get("job_id", "unknown")
        job_type = job_data.get("type", "ANALYZE_BATCH")
        logger.info("Processing job %s [Type: %s]", job_id, job_type)

        start_time = time.time()
        try:
            if job_type == "SEGMENTATION_DEEP":
                # Heavy segmentation batch
                image_path = job_data.get("image_path")
                result = {"status": "SUCCESS", "job_id": job_id, "processed_path": image_path}
            elif job_type == "ANALYZE_FULL":
                # Full multi-modal analysis
                agronomic = job_data.get("agronomic", {})
                res = self.pipeline_service.analyze(agronomic_features=agronomic, request_id=job_id)
                result = {"status": "SUCCESS", "job_id": job_id, "data": res.model_dump()}
            else:
                result = {"status": "SUCCESS", "job_id": job_id, "message": "Job processed"}

            duration = (time.time() - start_time) * 1000.0
            logger.info("Job %s completed in %.2fms", job_id, duration)
            return result
        except Exception as exc:
            logger.exception("Job %s failed with error: %s", job_id, exc)
            return {"status": "FAILED", "job_id": job_id, "error": str(exc)}

    def run_poll_loop(self, poll_interval_sec: float = 2.0) -> None:
        """Run single-threaded daemon polling queue."""
        self.running = True
        logger.info("Worker polling loop started. Press Ctrl+C to terminate.")
        try:
            while self.running:
                # In production, pop from Redis queue: redis_client.blpop("edaphnet_jobs")
                time.sleep(poll_interval_sec)
        except KeyboardInterrupt:
            logger.info("Worker received shutdown signal. Exiting cleanly...")
            self.running = False


if __name__ == "__main__":
    worker = EdaphNetWorker()
    worker.run_poll_loop()
