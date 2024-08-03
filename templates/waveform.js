let audioContext;
let analyser;
let dataArray;
let bufferLength;
let canvas;
let canvasCtx;
let mediaRecorder;
let source;
let stream;
let isRecording = false;
let chunks = [];
let recordingLength = 0;

document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('waveform');
    canvasCtx = canvas.getContext('2d');
});

async function toggleRecording() {
    await getRecordingLength();
    startRecording();
}

async function startRecording() {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 2048;
    bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
        chunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        chunks = [];
        const file = new File([blob], 'user_input.wav');
        const formData = new FormData();
        formData.append('file', file);

        await fetch('/upload_audio', {
            method: 'POST',
            body: formData,
        });

        stream.getTracks().forEach(track => track.stop());
        console.log('Recording stopped and sent to server');
    };

    mediaRecorder.start();

    setTimeout(() => {
        mediaRecorder.stop();
    }, recordingLength * 1000);

    isRecording = true;
    draw();
}

function draw() {
    if (!isRecording) {
        return;
    }
    requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = 'black';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'white';

    canvasCtx.beginPath();

    let sliceWidth = canvas.width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        let v = dataArray[i] / 128.0;
        let y = v * canvas.height / 2;

        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}