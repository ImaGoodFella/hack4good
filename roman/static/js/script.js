
document.getElementById("image_file_input").addEventListener("change", function(e) {
    console.log("image input change")
    var reader = new FileReader();
    reader.onload = function(e) {
        img = document.getElementById("image_preview");
        img.src = e.target.result;
    }
    reader.readAsDataURL(this.files[0]);
});


function form_element_to_formData_entry(f_e) {

}

document.getElementById("upload_button").addEventListener("click", async function(e) {
    e.preventDefault()
    //let image = document.getElementById("image_file_input").files[0];
    let form = document.getElementById("upload_form");
    let formData = new FormData(form);
    let response = await fetch(form.action, {method: 'post', body: formData})
    console.log(response)
    let data = await response.json()

    document.getElementById("csv_date").innerHTML = data["date"]
    document.getElementById("csv_lon").innerHTML = data["lon"]
    document.getElementById("csv_lat").innerHTML = data["lat"]
    document.getElementById("farmer_id").innerHTML = data["farmer_id"]
    document.getElementById("site_id").innerHTML = data["site_id"]

    document.getElementById("damage_extent").innerHTML = data["damage_extent"]+"%"
    document.getElementById("damage_type").innerHTML = data["damage_type"]
});