1. Download `cloudflared-windows-amd64.exe` from Cloudflare docs.
2. Rename to `cloudflared.exe` and place it in a folder on your `PATH`.
3. Verify:

```powershell
cloudflared --version
```

1. Start backend:

```powershell
cd d:\dev\stadtmuseum-soi25\task2\backend
python .\api3.py
```

2. Start quick tunnel (new terminal):

```powershell
cloudflared tunnel --url http://localhost:8000
```

3. cloudflared prints a public URL like `https://random-name.trycloudflare.com`.

4. Set backend public URL to that value before launching backend (or relaunch
   backend after setting):

```powershell
$env:PUBLIC_BASE_URL="https://random-name.trycloudflare.com"
python .\api3.py
```
