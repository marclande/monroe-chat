"""Inject OG meta tags into Streamlit's index.html at startup."""
import streamlit
import os

OG_TAGS = """
    <!-- OG Meta Tags -->
    <meta property="og:title" content="Explore Declassified Consciousness Archives" />
    <meta property="og:description" content="Real transcripts from Monroe Institute research, CIA Gateway studies, and military sessions." />
    <meta property="og:image" content="https://chatv2.lunabus.co/_stcore/static/og-preview.png" />
    <meta property="og:url" content="https://chatv2.lunabus.co" />
    <meta property="og:type" content="website" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="Explore Declassified Consciousness Archives" />
    <meta name="twitter:description" content="Real transcripts from Monroe Institute research, CIA Gateway studies, and military sessions." />
    <meta name="twitter:image" content="https://chatv2.lunabus.co/_stcore/static/og-preview.png" />
"""

def inject():
    index_path = os.path.join(os.path.dirname(streamlit.__file__), "static", "index.html")

    with open(index_path, "r") as f:
        html = f.read()

    if "og:title" in html:
        print("OG tags already injected, skipping.")
        return

    html = html.replace("<title>Streamlit</title>",
                        f"<title>Monroe Archives</title>\n{OG_TAGS}")

    with open(index_path, "w") as f:
        f.write(html)

    print(f"OG tags injected into {index_path}")

if __name__ == "__main__":
    inject()
