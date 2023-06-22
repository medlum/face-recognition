var socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port, {
    transports: ['websocket']
});
socket.on('connect', function () {
    console.log("Connected...!", socket.connected)
});


var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
const videoStream = document.querySelector("#videoElement");


videoStream.width = 400;
videoStream.height = 300;


if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({
        video: true
    })
        .then(function (stream) {
            videoStream.srcObject = stream;
            videoStream.play();
        })
        .catch(function (err0r) {
       });
}
const FPS = 5;
setInterval(() => {
    width = videoStream.width;
    height = videoStream.height;
    context.drawImage(videoStream, 0, 0, width, height);
    var data = canvas.toDataURL('image/jpeg', 0.5);
    context.clearRect(0, 0, width, height);
    socket.emit('image', data);
}, 1000 / FPS);

socket.on('processed_image', function (image) {
    photo.setAttribute('src', image); 
});