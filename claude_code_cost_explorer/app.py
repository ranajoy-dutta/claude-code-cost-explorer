"""Claude Code Cost Tracker. Run: ccx"""

import pathlib
from flask import Flask
from .utils import register_filters
from .routes import bp

app = Flask(__name__, template_folder=str(pathlib.Path(__file__).parent / "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True

register_filters(app)
app.register_blueprint(bp)


def main():
    import webbrowser
    import threading
    import os
    import argparse
    from waitress import serve

    parser = argparse.ArgumentParser(description="Claude Code Cost Explorer")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5050)))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    args = parser.parse_args()

    browser_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    url = f"http://{browser_host}:{args.port}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"  Claude Code Cost Explorer running at {url}")
    print(f"  Listening on {args.host}:{args.port}")
    print("  Press Ctrl+C to quit.")

    serve(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
