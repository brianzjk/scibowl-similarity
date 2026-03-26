from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scibowl.dedupe.review import DuplicateReviewStore, dump_json


def run_duplicate_review_server(
    *,
    candidates_path: Path,
    questions_path: Path,
    output_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    title: str = "Duplicate Review",
) -> None:
    store = DuplicateReviewStore(candidates_path, questions_path, output_path)
    html = (Path(__file__).with_name("review_app.html")).read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._respond_html(html.replace("__APP_TITLE__", title))
                return
            if parsed.path == "/api/summary":
                self._respond_json(store.summary())
                return
            if parsed.path == "/api/session":
                query = parse_qs(parsed.query)
                filter_name = query.get("filter", ["unreviewed"])[0]
                min_similarity = float(query.get("min_similarity", ["0"])[0] or "0")
                self._respond_json(
                    {
                        "items": store.session_items(filter_name=filter_name, min_similarity=min_similarity),
                        "summary": store.summary(),
                    }
                )
                return
            if parsed.path.startswith("/api/candidate/"):
                pair_id = parsed.path.removeprefix("/api/candidate/")
                self._respond_json(store.candidate_payload(pair_id))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            updated = store.save_review(
                pair_id=payload["pair_id"],
                label=payload.get("label"),
                notes=payload.get("notes"),
            )
            self._respond_json({"candidate": updated, "summary": store.summary()})

        def log_message(self, format: str, *args: object) -> None:
            return

        def _respond_html(self, payload: str) -> None:
            data = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _respond_json(self, payload: dict[str, object]) -> None:
            data = dump_json(payload)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving duplicate review app at http://{host}:{port}")
    print(f"Candidate source: {candidates_path}")
    print(f"Review output: {output_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
