
// ==============================
// GET HTML ELEMENTS
// ==============================

const fileInput =
    document.getElementById("fileInput");

const previewContainer =
    document.getElementById("previewContainer");

const previewImage =
    document.getElementById("previewImage");

const fileName =
    document.getElementById("fileName");

const predictButton =
    document.getElementById("predictButton");

const loading =
    document.getElementById("loading");

const result =
    document.getElementById("result");

const flowerName =
    document.getElementById("flowerName");

const confidence =
    document.getElementById("confidence");

const errorBox =
    document.getElementById("error");


// ==============================
// STORE SELECTED IMAGE
// ==============================

let selectedFile = null;


// ==============================
// IMAGE SELECTION
// ==============================


// ⭐ CHANGED:
// JavaScript now listens for the file input's
// "change" event.
//
// What actually changed:
// When the user selects an image,
// this code receives that image,
// creates a preview and enables
// the Identify Flower button.

fileInput.addEventListener(
    "change",
    function () {

        // Get selected file
        selectedFile =
            fileInput.files[0];


        // Stop if no file was selected
        if (!selectedFile) {

            return;

        }


        // ==============================
        // CHECK FILE TYPE
        // ==============================

        if (
            !selectedFile.type.startsWith(
                "image/"
            )
        ) {

            showError(
                "Please select a JPG, JPEG or PNG image."
            );

            selectedFile = null;

            predictButton.disabled =
                true;

            return;

        }


        // ==============================
        // CREATE PREVIEW
        // ==============================

        const imageURL =
            URL.createObjectURL(
                selectedFile
            );


        previewImage.src =
            imageURL;


        previewContainer.style.display =
            "block";


        // Show selected filename
        fileName.textContent =
            selectedFile.name;


        // Enable prediction button
        predictButton.disabled =
            false;


        // Clear previous result
        result.style.display =
            "none";

        errorBox.style.display =
            "none";

    }
);


// ==============================
// PREDICT FLOWER
// ==============================


// ⭐ CHANGED:
// Added the frontend-to-FastAPI connection.
//
// What actually changed:
// When the user clicks the button,
// JavaScript sends the selected image
// to:
//
// POST /predict
//
// FastAPI then sends the prediction back.

predictButton.addEventListener(
    "click",
    async function () {

        // Make sure image exists
        if (!selectedFile) {

            showError(
                "Please select a flower image first."
            );

            return;

        }


        // ==============================
        // CREATE FORM DATA
        // ==============================

        const formData =
            new FormData();


        formData.append(
            "file",
            selectedFile
        );


        // ==============================
        // SHOW LOADING
        // ==============================

        loading.style.display =
            "block";


        result.style.display =
            "none";

        errorBox.style.display =
            "none";


        predictButton.disabled =
            true;


        try {

            // ==============================
            // SEND IMAGE TO FASTAPI
            // ==============================


            // ⭐ CHANGED:
            // Use "/predict" instead of
            // "http://127.0.0.1:8000/predict".
            //
            // What actually changed:
            // The website and API are now served
            // by the same FastAPI server.
            //
            // Therefore "/predict" automatically means:
            //
            // http://127.0.0.1:8000/predict

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",

                        body: formData
                    }
                );


            // ==============================
            // CHECK RESPONSE
            // ==============================

            if (!response.ok) {

                throw new Error(
                    "Prediction failed."
                );

            }


            // Convert response into JSON
            const data =
                await response.json();


            // ==============================
            // DISPLAY FLOWER
            // ==============================

            flowerName.textContent =
                data.flower;


            // ==============================
            // DISPLAY CONFIDENCE
            // ==============================

            confidence.textContent =
                data.confidence + "%";


            // Show result
            result.style.display =
                "block";

        }


        catch (error) {

            console.error(
                "Prediction error:",
                error
            );


            showError(
                "Unable to connect to the prediction server. Make sure FastAPI is running."
            );

        }


        finally {

            // Hide loading
            loading.style.display =
                "none";


            // Enable button again
            predictButton.disabled =
                false;

        }

    }
);


// ==============================
// ERROR FUNCTION
// ==============================

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.style.display =
        "block";

}

