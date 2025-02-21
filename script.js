function Upload(){
    const fileInput1 = document.getElementById("video");
    if (!fileInput1 || !fileInput1.files.length) {
        return alert("No file selected");
    }

    const uploadedFile = fileInput1.files[0];

    if (!uploadedFile.type.includes('video')) {
        return alert("Only Videos allowed");
    }
    document.getElementById("load").style.display="flex";
}
