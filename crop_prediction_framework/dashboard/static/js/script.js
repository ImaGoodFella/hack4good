function update_entries(json_data, id_prefix="") {
    for (const key in json_data) {
        if (json_data.hasOwnProperty(key)) {
            document.getElementById(id_prefix+key).innerHTML = json_data[key];
        }
    }
}

function update_preview_image() {
    const file = document.getElementById("image_file_input").files[0]
    if (file == null) {
        return
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        img = document.getElementById("image_preview");
        img.src = e.target.result;
    }
    reader.readAsDataURL(file);
}

document.getElementById("image_file_input").addEventListener("change", function(e) {
    update_preview_image()
});


document.getElementById("upload_button").addEventListener("click", async function(e) {
    e.preventDefault()
    update_preview_image()
    const error_iframe = document.getElementById("error_iframe")
    const error_doc = error_iframe.contentDocument || error_iframe.contentWindow.document
    console.log(error_doc)
    const upload_info_div = error_doc.body//error_doc.getElementById("upload_error")
    //const upload_info_div = document.getElementById("upload_error")
    //TODO add loading animation
    upload_info_div.innerHTML = "Processing..."
    const form = document.getElementById("upload_form");
    const formData = new FormData(form);
    const response = await fetch(form.action, {method: 'post', body: formData})
    console.log(response)
    const answer = await response
    if (!answer.ok) {
        error_message = await answer.text()
        //alert(error_message)
        upload_info_div.innerHTML = error_message
        return
    }
    data = await answer.json()
    upload_info_div.innerHTML = ""
    update_entries(data)
});