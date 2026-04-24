import os
import shutil
import tempfile
import zipfile
import secrets
import logging
import paramiko
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from data_manager import SessionManager

logger = logging.getLogger("SFTP_Uploader")

# Load environment variables from .env if present
load_dotenv()

def upload_share_to_sftp(ids: list, ttl_minutes: int = 15) -> dict:
    if not ids:
        raise ValueError("No image ids provided")

    token = secrets.token_urlsafe(24).replace("-", "").replace("_", "")

    sftp_host = os.environ.get("SFTP_HOST")
    sftp_user = os.environ.get("SFTP_USER")
    sftp_key = os.environ.get("SFTP_KEY_PATH")
    public_domain = os.environ.get("PUBLIC_DOMAIN", "lottereininger.tuebingen.ai")

    if not sftp_host or not sftp_user or not sftp_key:
        raise RuntimeError("SFTP_HOST, SFTP_USER, and SFTP_KEY_PATH must all be set in .env")

    html_filename = f"share_{token}.html"

    with tempfile.TemporaryDirectory() as tmpdir:
        share_dir = Path(tmpdir)
        local_files = []
        paths = []

        for idx, image_id in enumerate(ids, start=1):
            path = SessionManager.find_path_by_id(image_id)
            if not path or "compositions" not in path.parts:
                continue
            filename = f"img_{token}_{idx:02d}.png"
            dst = share_dir / filename
            shutil.copy2(path, dst)
            local_files.append(dst)
            paths.append(filename)

        if not paths:
            raise ValueError("No valid images found for sharing")

        # Build zip of all images
        zip_filename = f"imgs_{token}_all.zip"
        zip_path = share_dir / zip_filename
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename in paths:
                zf.write(share_dir / filename, arcname=filename)
        local_files.append(zip_path)

        # Build gallery HTML items
        items_html = "".join(
            f'<div class="item-card">'
            f'<img src="{filename}" alt="Shared image" class="share-image" />'
            f'<div class="item-actions">'
            f'<a class="btn btn-download" href="{filename}" download="{filename}">Download</a>'
            f'</div></div>'
            for filename in paths
        )

        # Load Tuebingen AI SVG logo
        svg_path = Path(__file__).parent.parent / "frontend" / "src" / "assets" / "logo-tueai.svg"
        svg_content = ""
        if svg_path.exists():
            with open(svg_path, "r", encoding="utf-8") as svg_file:
                svg_content = svg_file.read()

        html_content = (
            "<!doctype html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8" />\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
            "  <title>Your Artworks</title>\n"
            "  <style>\n"
            "    body { font-family: system-ui, sans-serif; background: #f3f4f6; padding: 20px; color: #1f2937; display: flex; flex-direction: column; min-height: 100vh; margin: 0; }\n"
            "    .container { max-width: 800px; margin: 0 auto; flex: 1; width: 100%; }\n"
            "    h1 { text-align: center; }\n"
            "    .gallery { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); margin-top: 2rem; }\n"
            "    .item-card { background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 10px; text-align: center; }\n"
            "    .share-image { width: 100%; height: auto; border-radius: 4px; }\n"
            "    .item-actions { margin-top: 10px; }\n"
            "    .btn { display: inline-block; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 14px; text-align: center; cursor: pointer; }\n"
            "    .btn-download { background-color: #3b82f6; color: white; }\n"
            "    .btn-download:hover { background-color: #2563eb; }\n"
            "    .btn-primary { background-color: #10b981; color: white; margin-top: 20px; display: block; }\n"
            "    .btn-primary:hover { background-color: #059669; }\n"
            "    .footer-note { text-align: center; margin-top: 2rem; font-size: 0.875rem; color: #6b7280; }\n"
            "    .logo-container { text-align: center; margin-top: 3rem; padding-bottom: 2rem; }\n"
            "    .logo-container svg { height: 40px; width: auto; opacity: 0.8; transition: opacity 0.2s; }\n"
            "    .logo-container a:hover svg { opacity: 1; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            '  <div class="container">\n'
            "    <h1>Your Artworks</h1>\n"
            f'    <a href="{zip_filename}" class="btn btn-primary" download>Download All Images (ZIP)</a>\n'
            '    <div class="gallery">\n'
            f"        {items_html}\n"
            "    </div>\n"
            '    <div class="footer-note">\n'
            "        These images will be automatically deleted after 15 minutes.\n"
            "    </div>\n"
            "  </div>\n"
            '  <div class="logo-container">\n'
            '    <a href="https://tuebingen.ai" target="_blank" rel="noopener noreferrer">\n'
            f"      {svg_content}\n"
            "    </a>\n"
            "  </div>\n"
            "</body>\n"
            "</html>"
        )

        html_path = share_dir / html_filename
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        local_files.append(html_path)

        # Upload all files via paramiko SFTP into a per-session subfolder
        logger.info(f"Uploading {len(local_files)} files to {sftp_host} via SFTP...")
        try:
            transport = paramiko.Transport((sftp_host, 22))
            pkey = paramiko.Ed25519Key.from_private_key_file(sftp_key)
            transport.connect(username=sftp_user, pkey=pkey)
            sftp = paramiko.SFTPClient.from_transport(transport)

            base_dir = "/public_html/images"
            session_dir = f"{base_dir}/{token}"

            # Ensure base dir exists
            try:
                sftp.stat(base_dir)
            except FileNotFoundError:
                sftp.mkdir(base_dir)

            # Create per-session subfolder
            sftp.mkdir(session_dir)

            for local_p in local_files:
                sftp.put(str(local_p), f"{session_dir}/{local_p.name}")

            sftp.close()
            transport.close()
            logger.info(f"Upload complete. Share URL: https://{public_domain}/images/{token}/{html_filename}")
        except Exception as e:
            logger.error(f"SFTP Upload failed: {e}")
            raise RuntimeError(f"SFTP Upload failed: {e}")

    share_url = f"https://{public_domain}/images/{token}/{html_filename}"
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ttl_minutes)

    return {
        "token": token,
        "share_url": share_url,
        "expires_at": expires_at.isoformat()
    }
