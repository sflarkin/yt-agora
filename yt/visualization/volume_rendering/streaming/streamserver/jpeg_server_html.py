import yt.visualization.volume_rendering.streaming.streamserver.jpegjs as jpegjs


def get_html():
      return jpegjs.get_html() + """
      <script type="text/javascript">
      if(typeof jQuery == 'undefined'){
            //-------------------------------------
            // Include JQuery if it hasn't been already
            document.write("\<script src='//ajax.googleapis.com/ajax/libs/jquery/1.2.6/jquery.min.js' type='text/javascript'>\<\/script>");
      }
      </script>
      
      <div id = "container" >

            <canvas id="render_ctx" style="background: #000000;" oncontextmenu="return false;">
                  Your browser does not support the HTML5 canvas tag.
            </canvas>

            <div id = "menu_container" >

            <!-- Menu: Lighting Values -->
            <div id = "lt_menu_container" style="background-color:rgba(0,0,0,0.5); padding: 5px; font-family: verdana;position:relative; color:    fff; float: left;">
             <p>
                  Density: <input type="text" style="width: 40px;border: solid 1px white;" id="density"></br>
                  Brightness: <input type="text" style="width: 40px;border: solid 1px white;" id="brightness"></br>
             </p>
             </div>

            <!-- Menu: Transfer Function Values -->
            <div id = "tf_menu_container" style="background-color:rgba(0,0,0,0.5); padding: 5px; font-family: verdana;color: fff;position:relative;  float: left;">
             <p>
                  tf Lower: <input type="text" style="width: 40px;border: solid 1px white;" id="tflow"></br>
                  tf Upper: <input type="text" style="width: 40px;border: solid 1px white;" id="tfup"></br>
                  tf Scale: <input type="text" style="width: 40px;border: solid 1px white;" id="tfscale"></br>
                  Colormap: <input type="text" style="width: 40px;border: solid 1px white;" id="tfcolormap"></br>
             </p>
             </div>

            <!-- Menu: RayCaster Values -->
            <div id = "rc_menu_container" style="background-color:rgba(0,0,0,0.5); padding: 5px; font-family: verdana;color: fff;position:relative;  float: left;">
             <p>
                  near clip: <input type="text" style="width: 40px;border: solid 1px white;" id="nclip"></br>
                  far  clip: <input type="text" style="width: 40px;border: solid 1px white;" id="fclip"></br>
                  samples: <input type="text" style="width: 40px;border: solid 1px white;" id="samples"></br>
             </p>
             </div>

            <!-- Menu: Screen Values -->
            <div id = "s_menu_container" style="background-color:rgba(0,0,0,0.5); padding: 5px; font-family: verdana;color:  fff;position:relative;  float: left;">
             <p>
                  width: <input type="text" style="width: 40px;border: solid 1px white;" id="sw"></br>
                  height: <input type="text" style="width: 40px;border: solid 1px white;" id="sh"></br>
             </p>
             </div>
             </div>


      </div>


      <script type="text/javascript">

      // --------------------
      // CudaAstroPy
      // --------------------

      //-------------------------------------
      // Initialize the canvas

      var canvas = document.getElementById('render_ctx');
      canvas.width = width;
      canvas.height = height;
      var context = canvas.getContext('2d');

      //-------------------------------------
      // Communication with jpeg server

      var connected = false;
      var framerate = 0;
      var frame_count = 0;
      var lastCalledTime = 0;
      context.fillStyle = 'white';
      var URL = (window.URL || window.webkitURL);

      var url_address = "ws://127.0.0.1:" + port + "/websocket";

      var ws = new WebSocket(url_address);
      ws.onopen = function() {
            console.log("Stream open");
            connected = true;
            send_mouse_data(100);
      };
      ws.onmessage = function (evt) {
            var jpeg = new JpegImage();
            jpeg.onload = function(){
                  var image_data = context.getImageData(0,0,width,height);
                  jpeg.copyToImageData(image_data);
                  context.putImageData(image_data,0,0);
                  context.fillText(framerate, 10,10);
                  context.fillText(["X: ", mouseX].join(), 10,20);
                  context.fillText(["Y: ", mouseY].join(), 10,30);
                  context.restore();
                  if(frame_count++ < 10){
                        frame_count = 0;
                        var delta = (new Date().getTime() - lastCalledTime)/1000;
                        lastCalledTime = new Date().getTime();
                        framerate = Math.ceil(1/delta);
                  }
            };
            jpeg.load(URL.createObjectURL(new Blob([evt.data], {type: "image/jpeg"})));
      };
      ws.onclose = function(){
            connected = false;
            context.fillStyle = "rgba(0,0,0, 0.5)";
            context.rect(0,0,width,height);
            context.fill();
            context.fillStyle = 'white';
            context.textAlign = 'center';
            context.font = '20pt Arial';
            context.fillText("Connection Closed", width / 2, height / 2);
      }

      //-------------------------------------
      // Mouse events

      var dragging = false;
      var drag_counter = 0;
      var drag_sample = 3;
      var shifted = false;
      var shift_begin = true;
      var menu_focused = false;
      var mouseX = 0;
      var mouseY = 0;
      var orig_mouseX = 0;
      var orig_mouseY = 0;
      var prev_mouseX = 0;
      var prev_mouseY = 0;
      var mouse_button = '';
      $('canvas').mousedown(function(e){
            switch(e.which){
                  case 1 : mouse_button = 'L'; break;
                  case 2 : mouse_button = 'M'; break;
                  case 3 : mouse_button = 'R'; break;
                  defualt : mouse_button = 'L'; break;
            }
            dragging = true;
            menu_focused = false;
            orig_mouseX = e.pageX;
            orig_mouseY = e.pageY;
      });

      $('canvas').mouseup(function(e){
            dragging = false;
            mouse_button = 'L';
            send_mouse_data(100);
      });

      $('canvas').mousemove(function(e){
            var rect = render_ctx.getBoundingClientRect();
            /*
            prev_mouseX = mouseX;
            prev_mouseY = mouseY;
            mouseX = Math.floor(e.clientX - rect.left);
            mouseY = height - Math.floor(e.clientY - rect.top);
            if(mouseX > width) mouseX = width;
            if(mouseY > height) mouseY = height;
            */
            if(drag_counter++ === drag_sample){
                  drag_counter = 0;
                  prev_mouseX = mouseX;
                  prev_mouseY = mouseY;
                  mouseX = Math.floor(e.clientX - rect.left);
                  mouseY = height - Math.floor(e.clientY - rect.top);
                  if(mouseX > width) mouseX = width;
                  if(mouseY > height) mouseY = height;
                  if(dragging && !shifted) send_mouse_data(framerate);
                  if(shifted) {
                        if(shift_begin){
                              prev_image_data = context.getImageData(0,0,width,height);
                              shift_begin = false;
                        }
                        pick();
                  } else shift_begin = true;
            }

            if(mouseY < 150 && !dragging) $("#menu_container").fadeIn();
            if(mouseY > 150 && !menu_focused) $("#menu_container").fadeOut();

            if(mouseY < 150 && !dragging) $("#tf_menu_container").fadeIn();
            if(mouseY > 150 && !menu_focused) $("#tf_menu_container").fadeOut();

            if(mouseY < 150 && !dragging) $("#s_menu_container").fadeIn();
            if(mouseY > 150 && !menu_focused) $("#s_menu_container").fadeOut();

            if(mouseY < 150 && !dragging) $("#rc_menu_container").fadeIn();
            if(mouseY > 150 && !menu_focused) $("#rc_menu_container").fadeOut();
      });

      $('#density').keypress(function(){
            var tmp = document.getElementById("density").value;
            if(tmp != '') density = tmp;
            send_mouse_data(100);
      });

      $('#density').focus(function(){
            menu_focused = true;
      });

      $('#brightness').keypress(function(){
            var tmp = document.getElementById("brightness").value;
            if(tmp != '') brightness = tmp;
            send_mouse_data(100);
      });

      $('#brightness').focus(function(){
            menu_focused = true;
      });

      $('#tflow').keypress(function(){
            var tmp = document.getElementById("tflow").value;
            if(tmp != '') tflow = tmp;
            send_mouse_data(100);
      });

      $('#tflow').focus(function(){
            menu_focused = true;
      });

      $('#tfup').keypress(function(){
            var tmp = document.getElementById("tfup").value;
            if(tmp != '') tfup = tmp;
            send_mouse_data(100);
      });

      $('#tfup').focus(function(){
            menu_focused = true;
      });

      $('#tfcolormap').keypress(function(){
            var tmp = document.getElementById("tfcolormap").value;
            if(tmp != '') tfcolormap = tmp;
            send_mouse_data(100);
      });

      $('#tfcolormap').focus(function(){
            menu_focused = true;
      });


      function send_mouse_data(q){
            if(connected) ws.send([q, (prev_mouseX - mouseX), (prev_mouseY - mouseY), width, height, mouse_button, density, brightness , tflow, tfup, tfcolormap, (shifted)?1:0, picking_radius, mouseX, mouseY].join(" "));
      }

      //-------------------------------------
      // Menu
      var density     = 0.05;
      var brightness  = 1.0;

      var tflow      = 0.5;
      var tfup       = 1.0;
      var tfscale    = 1.0;
      var tfcolormap = 1;


      var nclip   = 0.01;
      var fclip   = 1.0;
      var samples = 10.0;

      var sw = 512;
      var sh = 512;

      var picking_radius = 10;
      var prev_image_data;
      $(document).bind('keyup keydown', function(e){shifted = e.shiftKey} );
      function pick(){
            if(mouse_button == 'M'){
                  picking_radius += (prev_mouseY - mouseY);
            }
            context.putImageData(prev_image_data,0,0);
            context.beginPath();
            context.arc(mouseX, height-mouseY, picking_radius, 0, 2 * Math.PI, false);
            context.lineWidth = 1;
            context.strokeStyle = 'white';
            context.stroke();
            if(dragging && mouse_button === 'L') send_mouse_data(100);
      }


      //Lighting values
      document.getElementById("density").defaultValue    = density; 
      document.getElementById("brightness").defaultValue = brightness;

      //Transfer Function Values
      document.getElementById("tflow").defaultValue      =  tflow;
      document.getElementById("tfup").defaultValue       =  tfup;
      document.getElementById("tfscale").defaultValue    =  tfscale;
      document.getElementById("tfcolormap").defaultValue =  tfcolormap;

      //RayCaster Values
      document.getElementById("nclip").defaultValue   =  nclip;
      document.getElementById("fclip").defaultValue   =  fclip;
      document.getElementById("samples").defaultValue =  samples;

      //RayCaster Values
      document.getElementById("sw").defaultValue =  sw;
      document.getElementById("sh").defaultValue =  sh;

      </script>
      """
