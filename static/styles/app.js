(() => {
  const width = 1280; // We will scale the photo width to this
  let height = 0; // This will be computed based on the input stream

  // |streaming| indicates whether or not we're currently streaming
  // video from the camera. Obviously, we start at false.

  let streaming = false;
  let data = null;
  let video = null;
  let canvas = null;
  let photo = null;
  let startbutton = null;

  function showViewLiveResultButton() {
    if (window.self !== window.top) {
      // Ensure that if our document is in a frame, we get the user
      // to first open it in its own tab or window. Otherwise, it
      // won't be able to request permission for camera access.
      document.querySelector(".contentarea").remove();
      const button = document.createElement("button");
      button.textContent = "View live result of the example code above";
      document.body.append(button);
      button.addEventListener("click", () => window.open(location.href));
      return true;
    }
    return false;
  }

  function startup() {
    if (showViewLiveResultButton()) {
      return;
    }
    video = document.getElementById("video");
    canvas = document.getElementById("canvas");
    photo = document.getElementById("photo");
    startbutton = document.getElementById("startbutton");

    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      .then((stream) => {
        video.srcObject = stream;
        video.play();
      })
      .catch((err) => {
        console.error(`An error occurred: ${err}`);
      });

    video.addEventListener(
      "canplay",
      (ev) => {
        if (!streaming) {
          height = video.videoHeight / (video.videoWidth / width);

          if (isNaN(height)) {
            height = width / (16 / 9);
          }

          video.setAttribute("width", width);
          video.setAttribute("height", height);
          canvas.setAttribute("width", width);
          canvas.setAttribute("height", height);
          streaming = true;
        }
      },
      false,
    );

    startbutton.addEventListener(
        "click",
        (ev) => {
            ev.preventDefault(); // Prevent the default form submission behavior
            // Call upload_image first
            uploadImage().then(() => {
            // After upload_image is done, submit the form
            document.getElementById("captureForm").submit();
            });
        },
        false
    );

    clearphoto();
  }

  // Fill the photo with an indication that none has been
  // captured.

  function clearphoto() {
    const context = canvas.getContext("2d");
    context.fillStyle = "#AAA";
    context.fillRect(0, 0, canvas.width, canvas.height);

    const data = canvas.toDataURL("image/png");
  }

  async function uploadImage() {
  return new Promise((resolve, reject) => {
    takepicture(); // Assuming takepicture handles the image capture logic
    console.log(data)
    // debugger;
    fetch('/upload_image', {
      method: 'POST',
      body: data,
      headers: {
        'Content-Type': 'application/octet-stream'
      }
    })
    .then(response => response.json())
    .then(data => {
      console.log(data.message);
      resolve(); // Resolve the promise when upload is successful
    })
    .catch(error => {
      console.error('Error:', error);
      reject(error); // Reject the promise if there's an error
    });
  });
    }

  // Capture a photo by fetching the current contents of the video
  // and drawing it into a canvas, then converting that to a PNG
  // format data URL. By drawing it on an offscreen canvas and then
  // drawing that to the screen, we can change its size and/or apply
  // other changes before drawing it.

  function takepicture() {
    const context = canvas.getContext("2d");
    if (width && height) {
      canvas.width = width;
      canvas.height = height;
      context.drawImage(video, 0, 0, width, height);

      data = canvas.toDataURL("image/png");
    //   debugger;
      console.log(data)
    } else {
      clearphoto();
    }
  }

  // Set up our event listener to run the startup process
  // once loading is complete.
  window.addEventListener("load", startup, false);
})();