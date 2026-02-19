document.getElementById("predictionForm").onsubmit = async function(e){

    e.preventDefault();
    
    const features = [
    
    +age.value,
    +sex.value,
    +cp.value,
    +trestbps.value,
    +chol.value,
    +fbs.value,
    +restecg.value,
    +thalach.value,
    +exang.value,
    +oldpeak.value,
    +slope.value,
    +ca.value,
    +thal.value
    
    ];
    
    const response = await fetch("/predict",{
    
    method:"POST",
    headers:{ "Content-Type":"application/json"},
    body:JSON.stringify({features})
    
    });
    
    const result = await response.json();
    
    document.getElementById("resultCard").classList.remove("hidden");
    
    document.getElementById("predictionText").innerText =
    "Prediction: " + result.prediction;
    
    if(result.confidence){
    
    document.getElementById("confidenceText").innerText =
    "Confidence: " + result.confidence + "%";
    
    }
    
    }
    
    function fillSample(){
    
    age.value=52
    sex.value=1
    cp.value=0
    trestbps.value=125
    chol.value=212
    fbs.value=0
    restecg.value=1
    thalach.value=168
    exang.value=0
    oldpeak.value=1.2
    slope.value=1
    ca.value=0
    thal.value=2
    
    }
    