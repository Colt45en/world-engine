paste into your merged file

Search for // === BEGIN EMBEDDED SERVER === and // === END EMBEDDED SERVER === to see the block you’re adding. By default we use SSE (no new libs). If you want true WebSocket, flip the macro and link websocketpp + ASIO (notes inline).
// === BEGIN EMBEDDED SERVER ===
#define VEC_SERVER_PORT 7777
#define ENABLE_WEBSOCKETPP 0   // set to 1 if you add websocketpp + ASIO
#define ENABLE_SSE_SERVER  1   // no extra deps, default path

#include <atomic>
#include <thread>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <condition_variable>
#include <cstring>
#include <sys/types.h>
#ifdef _WIN32
  #include <winsock2.h>
  #pragma comment(lib, "ws2_32.lib")
  using socklen_t = int;
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

// ---- shared snapshot the server reads ----
struct BrainSnapshot {
    BrainIO in{};
    BrainIO out{};
    int frame{0};
} g_snapshot;

std::mutex g_snap_mtx;

static std::string html_dashboard() {
    std::ostringstream o;
    o <<
R"(<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Nucleus Dashboard</title>
<style>
body{margin:0;background:#0f1116;color:#e6e6e6;font-family:ui-monospace,monospace;}
header{padding:12px 16px;border-bottom:1px solid #2a2f3a;background:#131621}
.wrap{display:flex;height:calc(100vh - 48px);}
#left{width:42%;border-right:1px solid #2a2f3a;height:100%;}
#right{flex:1;padding:12px;}
iframe{border:0;width:100%;height:100%}
.stat{font-size:12px;opacity:.8}
canvas{width:100%;height:100%}
</style>
</head>
<body>
<header>
  <strong>🧠 Nucleus Live</strong>
  <span class="stat" id="status"> • connecting…</span>
</header>
<div class="wrap">
  <div id="left">
    <iframe src="/panel"></iframe>
  </div>
  <div id="right">
    <canvas id="chart"></canvas>
  </div>
</div>
<script>
const statusEl = document.getElementById('status');
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
let W,H; function resize(){ W=canvas.width=canvas.clientWidth; H=canvas.height=canvas.clientHeight; }
window.addEventListener('resize', resize); resize();

let t=0, buf=[];
function push(v){ buf.push(v); if(buf.length>512) buf.shift(); }

function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#e6e6e6'; ctx.font='12px ui-monospace';
  ctx.fillText('o_toggle_spot, o_spawn_note, o_toggle_cube, o_heart_pulse', 8, 16);

  const names=['spot','note','cube','pulse'];
  const colors=['#9ef','#fe9','#9f9','#f99'];
  for(let j=0;j<4;j++){
    ctx.beginPath();
    for(let i=0;i<buf.length;i++){
      const v = buf[i][j] || 0;
      const x = (i/(buf.length-1))* (W-20)+10;
      const y = H - 20 - v*(H-60);
      if(i==0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.strokeStyle=colors[j]; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle=colors[j]; ctx.fillText(names[j], W-80, 20+14*j);
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

// Prefer WebSocket if available; else fallback to SSE
function startWS(){
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = ()=> statusEl.textContent=' • ws connected';
  ws.onclose = ()=> { statusEl.textContent=' • ws closed → fallback sse'; startSSE(); };
  ws.onerror = ()=> { statusEl.textContent=' • ws error → fallback sse'; ws.close(); };
  ws.onmessage = (ev)=>{
    try{
      const s=JSON.parse(ev.data);
      push([s.out.o_toggle_spot, s.out.o_spawn_note, s.out.o_toggle_cube, s.out.o_heart_pulse]);
    }catch(e){}
  };
}

function startSSE(){
  const es = new EventSource('/sse');
  es.onopen = ()=> statusEl.textContent=' • sse connected';
  es.onerror= ()=> statusEl.textContent=' • sse error';
  es.onmessage = (e)=>{
    try{
      const s=JSON.parse(e.data);
      push([s.out.o_toggle_spot, s.out.o_spawn_note, s.out.o_toggle_cube, s.out.o_heart_pulse]);
    }catch(_){}
  };
}
startWS();
</script>
</body></html>)";
    return o.str();
}

static std::string html_panel() {
    return R"(<!doctype html><html><head><meta charset="utf-8"/>
<style>body{margin:0;background:#0f1116;color:#e6e6e6;font-family:ui-monospace,monospace}
.wrap{padding:12px} .row{margin:6px 0} .k{opacity:.7}</style>
<title>Panel</title></head><body><div class="wrap">
<h3>Brain Panel</h3>
<div class="row k">Live snapshot from app (updates via fetch):</div>
<pre id="pre">{}</pre>
<script>
async function tick(){
  try{ const r=await fetch('/state'); const j=await r.json();
        document.getElementById('pre').textContent=JSON.stringify(j,null,2); }
  catch(_){}
  setTimeout(tick, 250);
}
tick();
</script>
</div></body></html>)";
}

// ------------- Minimal HTTP + SSE -------------
static std::string http_ok(const std::string& body, const char* ctype="text/html"){
    std::ostringstream o;
    o << "HTTP/1.1 200 OK\r\nContent-Type: " << ctype
      << "\r\nContent-Length: " << body.size()
      << "\r\nConnection: close\r\n\r\n" << body;
    return o.str();
}
static std::string http_notfound(){
    return "HTTP/1.1 404 Not Found\r\nContent-Length:0\r\nConnection: close\r\n\r\n";
}

#if ENABLE_SSE_SERVER
class SSEBroker {
public:
    void broadcast(const std::string& data){
        std::lock_guard<std::mutex> lk(m_);
        for (auto it = conns_.begin(); it != conns_.end();){
#ifdef _WIN32
            int sent = send(*it, data.c_str(), (int)data.size(), 0);
#else
            ssize_t sent = ::send(*it, data.c_str(), data.size(), MSG_NOSIGNAL);
#endif
            if (sent <= 0) {
#ifdef _WIN32
                closesocket(*it);
#else
                ::close(*it);
#endif
                it = conns_.erase(it);
            } else ++it;
        }
    }
    void add(int sock){
        std::lock_guard<std::mutex> lk(m_);
        conns_.push_back(sock);
    }
private:
    std::mutex m_;
    std::vector<int> conns_;
};
static SSEBroker g_sse;

static void sse_handshake(int client){
    const char* hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n\r\n";
#ifdef _WIN32
    send(client, hdr, (int)strlen(hdr), 0);
#else
    ::send(client, hdr, strlen(hdr), MSG_NOSIGNAL);
#endif
    g_sse.add(client);
}
#endif

static std::string snapshot_json(){
    std::lock_guard<std::mutex> lk(g_snap_mtx);
    const auto& s = g_snapshot;
    json j;
    j["frame"] = s.frame;
    auto toJ = [](const BrainIO& x){
        return json{
            {"t_sin",x.t_sin},{"t_cos",x.t_cos},{"frame_norm",x.frame_norm},
            {"heart_norm",x.heart_norm},{"spotlight_on",x.spotlight_on},{"avg_cube_dist",x.avg_cube_dist},
            {"o_toggle_spot",x.o_toggle_spot},{"o_spawn_note",x.o_spawn_note},
            {"o_toggle_cube",x.o_toggle_cube},{"o_heart_pulse",x.o_heart_pulse}
        };
    };
    j["in"]  = toJ(s.in);
    j["out"] = toJ(s.out);
    return j.dump();
}

static void http_server_loop(std::atomic<bool>& runflag){
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
#endif
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return;
    int opt=1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_port=htons(VEC_SERVER_PORT); addr.sin_addr.s_addr=htonl(INADDR_ANY);
    if (bind(srv, (sockaddr*)&addr, sizeof(addr))<0) {
#ifdef _WIN32
        closesocket(srv); WSACleanup();
#else
        ::close(srv);
#endif
        return;
    }
    listen(srv, 16);

    while (runflag){
        sockaddr_in caddr{}; socklen_t clen=sizeof(caddr);
        int client = accept(srv, (sockaddr*)&caddr, &clen);
        if (client < 0) continue;

        // Read a small request
        char buf[2048]; int n=recv(client, buf, sizeof(buf)-1, 0);
        if (n<=0){
#ifdef _WIN32
            closesocket(client);
#else
            ::close(client);
#endif
            continue;
        }
        buf[n]=0;
        std::string req(buf);

        auto pathBeg = req.find(' ');
        auto pathEnd = req.find(' ', pathBeg+1);
        std::string path = (pathBeg!=std::string::npos && pathEnd!=std::string::npos) ? req.substr(pathBeg+1, pathEnd-pathBeg-1) : "/";

        if (path == "/") {
            auto b = html_dashboard(); auto res=http_ok(b,"text/html");
#ifdef _WIN32
            send(client, res.c_str(), (int)res.size(), 0); closesocket(client);
#else
            ::send(client, res.c_str(), res.size(), MSG_NOSIGNAL); ::close(client);
#endif
        } else if (path == "/panel") {
            auto b = html_panel(); auto res=http_ok(b,"text/html");
#ifdef _WIN32
            send(client, res.c_str(), (int)res.size(), 0); closesocket(client);
#else
            ::send(client, res.c_str(), res.size(), MSG_NOSIGNAL); ::close(client);
#endif
        } else if (path == "/state") {
            auto b = snapshot_json(); auto res=http_ok(b,"application/json");
#ifdef _WIN32
            send(client, res.c_str(), (int)res.size(), 0); closesocket(client);
#else
            ::send(client, res.c_str(), res.size(), MSG_NOSIGNAL); ::close(client);
#endif
#if ENABLE_SSE_SERVER
        } else if (path == "/sse") {
            sse_handshake(client);
            // do NOT close; SSEBroker owns the socket now
#endif
        } else {
            auto res=http_notfound();
#ifdef _WIN32
            send(client, res.c_str(), (int)res.size(), 0); closesocket(client);
#else
            ::send(client, res.c_str(), res.size(), MSG_NOSIGNAL); ::close(client);
#endif
        }
    }

#ifdef _WIN32
    closesocket(srv); WSACleanup();
#else
    ::close(srv);
#endif
}

static std::thread g_httpThread;
static std::atomic<bool> g_httpRun{false};

static void start_http(){
    g_httpRun = true;
    g_httpThread = std::thread(http_server_loop, std::ref(g_httpRun));
}
static void stop_http(){
    g_httpRun = false;
    if (g_httpThread.joinable()) g_httpThread.join();
}

static void sse_broadcast_tick(){
#if ENABLE_SSE_SERVER
    // Send one SSE event with the full snapshot
    std::string data = snapshot_json();
    std::ostringstream o; o << "data: " << data << "\n\n";
    g_sse.broadcast(o.str());
#endif
}
// === END EMBEDDED SERVER ===
