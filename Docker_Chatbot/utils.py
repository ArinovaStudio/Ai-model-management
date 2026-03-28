# utils.py

def is_hinglish(orig: str, trans: str) -> bool:
    """Detects if the user's query contains Hinglish keywords."""
    kw = {"kya", "hai", "hain", "kaun", "kab", "kaise", "mera", "ko", "ne", "aur", "batao", "abhi", "aaj", "kal", "mein", "tha", "thi", "gaya", "haan", "nahi", "ki", "ke"}
    words = set(orig.lower().replace("?", "").replace(".", "").split())
    return bool(words.intersection(kw))

def get_html_ui() -> str:
    """Returns the HTML/JS for the text-only chatbot UI."""
    return """
    <!DOCTYPE html>
    <html>
    <head><style>
        body{font-family:'Segoe UI', sans-serif; background:#f4f7f6; display:flex; justify-content:center; padding-top:50px;} 
        .box{width:600px; background:white; padding:20px; border-radius:10px; box-shadow:0 4px 15px rgba(0,0,0,0.1);} 
        #log{height:400px; overflow-y:auto; border:1px solid #ddd; padding:15px; margin-bottom:15px; background:#f9fbfc; border-radius:5px;} 
        input{flex-grow:1; padding:12px; border:1px solid #ccc; border-radius:4px; font-size:15px;} 
        button{padding:12px 25px; background:#007bff; color:white; border:none; cursor:pointer; border-radius:4px; font-size:15px; transition:0.2s;}
        button:hover{background:#0056b3;}
        p {margin: 8px 0; line-height: 1.4;}
    </style></head>
    <body>
        <div class="box">
            <h2>💬 AI Office Chatbot</h2>
            <div id="log"></div>
            <div style="display:flex; gap:10px;">
                <input type="text" id="msg" placeholder="Type your message here..." autocomplete="off" onkeypress="if(event.key === 'Enter') sendMsg()"/>
                <button onclick="sendMsg()">Send</button>
            </div>
        </div>
        <script>
            let ws = new WebSocket("ws://" + window.location.host + "/ws/chat");
            let log = document.getElementById('log'), inp = document.getElementById('msg');
            ws.onmessage = e => { log.innerHTML += "<p>🤖 <b>Bot:</b> " + JSON.parse(e.data).reply + "</p>"; log.scrollTop = log.scrollHeight; };
            function sendMsg() { 
                if(!inp.value) return; 
                log.innerHTML += "<p>👤 <b>You:</b> " + inp.value + "</p>"; 
                ws.send(JSON.stringify({query: inp.value})); 
                inp.value = ''; 
                log.scrollTop = log.scrollHeight;
            }
        </script>
    </body>
    </html>
    """