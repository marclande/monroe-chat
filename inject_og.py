"""Inject OG meta tags into Streamlit's index.html at startup."""
import streamlit
import os
import traceback

OG_TAGS = """
    <!-- OG Meta Tags -->
    <meta property="og:title" content="Explore Declassified Consciousness Archives" />
    <meta property="og:description" content="Real transcripts from Monroe Institute research, CIA Gateway studies, and military sessions." />
    <meta property="og:image" content="https://raw.githubusercontent.com/marclande/monroe-chat/main/og-preview.jpg" />
    <meta property="og:url" content="https://chatv2.lunabus.co" />
    <meta property="og:type" content="website" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="Explore Declassified Consciousness Archives" />
    <meta name="twitter:description" content="Real transcripts from Monroe Institute research, CIA Gateway studies, and military sessions." />
    <meta name="twitter:image" content="https://raw.githubusercontent.com/marclande/monroe-chat/main/og-preview.jpg" />
"""

def inject():
    try:
        index_path = os.path.join(os.path.dirname(streamlit.__file__), "static", "index.html")
        print(f"Streamlit index.html path: {index_path}")
        print(f"File exists: {os.path.exists(index_path)}")
        print(f"File writable: {os.access(index_path, os.W_OK)}")

        with open(index_path, "r") as f:
            html = f.read()

        if "og:title" in html:
            print("OG tags already injected, skipping.")
            return

        html = html.replace("<title>Streamlit</title>",
                            f"<title>Monroe Archives</title>\n{OG_TAGS}")

        with open(index_path, "w") as f:
            f.write(html)

        # Verify
        with open(index_path, "r") as f:
            verify = f.read()
        if "og:title" in verify:
            print("SUCCESS: OG tags injected!")
        else:
            print("FAILED: OG tags not found after write")
    except Exception as e:
        print(f"ERROR injecting OG tags: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    inject()
