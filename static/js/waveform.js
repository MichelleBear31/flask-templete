let audioContext;
let analyser;
let dataArray;
let bufferLength;
let canvas;
let canvasCtx;
let stream;
let source;
let isRecording = false;

document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('waveform');
    if (canvas) {
        canvasCtx = canvas.getContext('2d');
        canvas.addEventListener('click', toggleRecording);
    } else {
        console.error('Canvas element not found');
    }
});

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize = 2048;
        bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);

        isRecording = true;
        draw();

        // 发送录音请求到服务器
        await fetch('/record_audio', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
            })
            .catch(error => {
                console.error('Error:', error);
            });
    } catch (err) {
        console.error('Error accessing media devices.', err);
    }
}

function draw() {
    if (!isRecording) {
        return;
    }
    requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = '#D4CCC2';
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

function stopRecording() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
    }
    isRecording = false;
}

// let audioContext;
// let analyser;
// let dataArray;
// let bufferLength;
// let canvas;
// let canvasCtx;
// let mediaRecorder;
// let source;
// let stream;
// let isRecording = false;

// document.addEventListener('DOMContentLoaded', () => {
//     canvas = document.getElementById('waveform');
//     canvasCtx = canvas.getContext('2d');
//     canvas.addEventListener('click', toggleRecording);
// });

// async function toggleRecording() {
//     if (isRecording) {
//         stopRecording();
//     } else {
//         await startRecording();
//     }
// }

// async function startRecording() {
//     stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//     audioContext = new (window.AudioContext || window.webkitAudioContext)();
//     analyser = audioContext.createAnalyser();
//     source = audioContext.createMediaStreamSource(stream);
//     source.connect(analyser);
//     analyser.fftSize = 2048;
//     bufferLength = analyser.frequencyBinCount;
//     dataArray = new Uint8Array(bufferLength);

//     isRecording = true;
//     draw();
    
//     mediaRecorder = new MediaRecorder(stream, {
//         mimeType: 'audio/wav',
//         audioBitsPerSecond: 44100 * 16,
//         bitsPerSample: 16,
//         numberOfChannels: 1,
//         frameRate: 1
//     });

//     // 定義停止錄音的函數
//     const stopRecordingAfterDuration = () => {
//         if (isRecording) {
//             stopRecording();
//             mediaRecorder.stop();
//         }
//     };

//     // 監聽錄音結束事件，並停止錄音
//     mediaRecorder.onstop = () => {
//         stopRecording();
//     };

//     // 錄音完成後的處理
//     mediaRecorder.ondataavailable = (event) => {
//         let audioBlob = event.data;
//         let formData = new FormData();
//         formData.append('audio_data', audioBlob, 'user_input.wav');

//         fetch('/save_audio', {
//             method: 'POST',
//             body: formData
//         }).then(response => response.json())
//         .then(data => {
//             console.log(data.message);
//         }).catch(error => {
//             console.error('Error:', error);
//         });
//     };

//     // 開始錄音
//     mediaRecorder.start();

//     // 3 秒後自動停止錄音
//     setTimeout(stopRecordingAfterDuration, 3000); // 3 seconds
// }

// function draw() {
//     if (!isRecording) {
//         return;
//     }
//     requestAnimationFrame(draw);

//     analyser.getByteTimeDomainData(dataArray);

//     canvasCtx.fillStyle = '#D4CCC2';
//     canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

//     canvasCtx.lineWidth = 2;
//     canvasCtx.strokeStyle = 'white';

//     canvasCtx.beginPath();

//     let sliceWidth = canvas.width * 1.0 / bufferLength;
//     let x = 0;

//     for (let i = 0; i < bufferLength; i++) {
//         let v = dataArray[i] / 128.0;
//         let y = v * canvas.height / 2;

//         if (i === 0) {
//             canvasCtx.moveTo(x, y);
//         } else {
//             canvasCtx.lineTo(x, y);
//         }

//         x += sliceWidth;
//     }

//     canvasCtx.lineTo(canvas.width, canvas.height / 2);
//     canvasCtx.stroke();
// }

// function stopRecording() {
//     if (stream) {
//         stream.getTracks().forEach(track => track.stop());
//     }
//     if (audioContext) {
//         audioContext.close();
//     }
//     isRecording = false;
// }