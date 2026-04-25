import json
import os

CLAUDE_DIR = os.path.expanduser("~/.claude")
projects_dir = os.path.join(CLAUDE_DIR, "projects")

# Find session 5f6d74e6
for encoded_name in os.listdir(projects_dir):
    proj_dir = os.path.join(projects_dir, encoded_name)
    if not os.path.isdir(proj_dir):
        continue
    for fname in os.listdir(proj_dir):
        if fname.startswith("5f6d74e6") and fname.endswith(".jsonl"):
            fpath = os.path.join(proj_dir, fname)
            print(f"File: {fpath}\n")
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    rtype = r.get("type")
                    uuid = r.get("uuid", "")[:12]
                    parent = (
                        r.get("parentUuid", "")[:12] if r.get("parentUuid") else "None"
                    )

                    if rtype == "assistant":
                        msg = r.get("message", {})
                        model = msg.get("model", "?")
                        content = msg.get("content", [])
                        usage = msg.get("usage", {})
                        content_types = []
                        for b in content:
                            if isinstance(b, dict):
                                btype = b.get("type", "?")
                                if btype == "text":
                                    content_types.append(
                                        f"text({len(b.get('text', ''))})"
                                    )
                                elif btype == "thinking":
                                    content_types.append(
                                        f"thinking({len(b.get('thinking', ''))})"
                                    )
                                elif btype == "tool_use":
                                    content_types.append(
                                        f"tool_use({b.get('name', '')})"
                                    )
                                else:
                                    content_types.append(btype)
                        in_tok = usage.get("input_tokens", 0)
                        out_tok = usage.get("output_tokens", 0)
                        print(
                            f"L{line_num} ASSISTANT uuid={uuid} parent={parent} model={model}"
                        )
                        print(f"  content: {content_types}")
                        print(f"  usage: in={in_tok} out={out_tok}")
                        print()
                    elif rtype == "user":
                        msg = r.get("message", {})
                        content = msg.get("content") if msg else r.get("content")
                        if isinstance(content, list):
                            types = [
                                it.get("type", "?") if isinstance(it, dict) else "str"
                                for it in content
                            ]
                        elif isinstance(content, str):
                            types = [f"str({len(content)})"]
                        else:
                            types = ["?"]
                        print(f"L{line_num} USER uuid={uuid} parent={parent}")
                        print(f"  content types: {types}")
                        print()
            break
