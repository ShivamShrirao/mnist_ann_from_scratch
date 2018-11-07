// set canvas id to variable
var canvas = document.getElementById("draw");

// get canvas 2D ctx and set it to the correct size
var ctx = canvas.getContext("2d");

// add event listeners to specify when functions should be triggered
document.addEventListener("mousemove", draw);
document.addEventListener("mousedown", setPosition);
document.addEventListener("mouseenter", setPosition);

// last known position
var pos = { x: 0, y: 0 };

// new position from mouse events
function setPosition(e) {
	pos.x = e.clientX;
	pos.y = e.clientY;
};

function draw(e) {
	if (e.buttons !== 1) return; // if mouse is pressed.....
	ctx.beginPath(); // begin the drawing path
	ctx.lineWidth = 8; // width of line
	ctx.lineCap = "round"; // rounded end cap
	ctx.moveTo(pos.x, pos.y); // from position
	setPosition(e);
	ctx.lineTo(pos.x, pos.y); // to position
	ctx.stroke(); // draw it!
	ctx.closePath();
};

function redraw(){
	ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // Clears the canvas
};

function submit(){
	var photo = canvas.toDataURL('image/png');
	var xhr = new XMLHttpRequest();
	xhr.open('POST', 'http://127.0.0.1:9050/handle.php', true);
	xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
	xhr.onload = function () {
    	console.log(this.responseText);
	};
	xhr.send("input="+photo);
	// var imgData=ctx.getImageData(0,0,canvas.width,canvas.height);
	// for (var i = 3; i < canvas.width*canvas.height*4; i+=4){
			// imgData.data[i]=100;
	// }
	// ctx.putImageData(imgData, 0, 0);
};

// function grayscale(){
// 	var imgData=ctx.getImageData(0,0,canvas.width,canvas.height);
//     for (var i = 0; i < imgData.data.length; i += 4) {
//       var avg = (imgData.data[i] + imgData.data[i + 1] + imgData.data[i + 2]) / 3;
//       imgData.data[i]     = avg; // red
//       imgData.data[i + 1] = avg; // green
//       imgData.data[i + 2] = avg; // blue
//     }
//     ctx.putImageData(imgData, 0, 0);
// };