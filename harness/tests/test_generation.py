from __future__ import annotations

import unittest
from unittest.mock import Mock

from promptbench.live.campaign import Campaign
from promptbench.live.generation import generation_metadata
from promptbench.live.transport import HTTPResponseError, response_json
from promptbench.runner import RunPaused
from promptbench.storage import IntegrityError


class GenerationTests(unittest.TestCase):
    def test_campaign_does_not_wait_past_its_budget(self):
        execution = Mock()
        execution.store.exists.return_value = False
        execution.remaining_seconds.return_value = 1
        campaign = Campaign(execution)
        from unittest.mock import patch

        with (
            patch("promptbench.live.campaign.normalize", return_value={"generation_id": "gen-123"}),
            patch.object(campaign, "metadata", side_effect=HTTPResponseError(404, None)),
            patch("promptbench.live.campaign.time.sleep") as sleep,
        ):
            with self.assertRaises(RunPaused):
                campaign.reconcile_generation("attempt", "dated")
        sleep.assert_not_called()

    def test_documented_url_auth_and_deferred_404_retry(self):
        get = Mock(
            side_effect=[
                HTTPResponseError(404, None),
                {"data": {"id": "gen-123", "model": "dated"}},
            ]
        )
        wait = Mock()
        self.assertEqual(generation_metadata(get, wait, "gen-123", "dated")["id"], "gen-123")
        self.assertEqual(get.call_count, 2)
        get.assert_called_with(
            "generation", "https://openrouter.ai/api/v1/generation?id=gen-123", authenticated=True
        )
        wait.assert_called_once_with(2)

    def test_retry_bound_and_auth_errors_do_not_loop(self):
        get, wait = Mock(side_effect=HTTPResponseError(404, None)), Mock()
        with self.assertRaises(HTTPResponseError):
            generation_metadata(get, wait, "gen-123", "dated")
        self.assertEqual(get.call_count, 5)
        self.assertEqual([c.args[0] for c in wait.call_args_list], [2, 4, 8, 16])
        get = Mock(side_effect=HTTPResponseError(401, None))
        with self.assertRaises(HTTPResponseError):
            generation_metadata(get, Mock(), "gen-123", "dated")
        self.assertEqual(get.call_count, 1)

    def test_identity_validation_and_http_status(self):
        for data in ({"id": "gen-other", "model": "dated"}, {"id": "gen-123", "model": "drift"}):
            with self.assertRaises(IntegrityError):
                generation_metadata(Mock(return_value={"data": data}), Mock(), "gen-123", "dated")
        get = Mock()
        with self.assertRaises(IntegrityError):
            generation_metadata(get, Mock(), "gen-123&other=1", "dated")
        get.assert_not_called()
        with self.assertRaises(HTTPResponseError) as caught:
            response_json({"http_status": 404, "error": None})
        self.assertEqual(caught.exception.status, 404)
