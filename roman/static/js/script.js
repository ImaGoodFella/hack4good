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
    const form = document.getElementById("upload_form");
    const formData = new FormData(form);
    const response = await fetch(form.action, {method: 'post', body: formData})
    console.log(response)
    const answer = await response
    if (!answer.ok) {
        alert(await answer.text())
        return
    }
    data = await answer.json()

    update_entries(data)
});