let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var sample1 = 0, sample2 = 0, sample3=0;
let isPredicting = false;
var skill_1 = document.getElementById('skill');
var timer;

$(document).ready(function () {
  $(window).scroll(function () {
      // sticky navbar on scroll script
      if (this.scrollY > 20) {
          $('.navbar').addClass("sticky");
      } else {
          $('.navbar').removeClass("sticky");
      }

      // scroll-up button show/hide script
      if (this.scrollY > 500) {
          $('.scroll-up-btn').addClass("show");
      } else {
          $('.scroll-up-btn').removeClass("show");
      }
  });

  // slide-up script
  $('.scroll-up-btn').click(function () {
      $('html').animate({ scrollTop: 0 });
      // removing smooth scroll on slide-up button click
      $('html').css("scrollBehavior", "auto");
  });

  $('.navbar .menu li a').click(function () {
      // applying again smooth scroll on menu items click
      $('html').css("scrollBehavior", "smooth");
  });

  // toggle menu/navbar script
  $('.menu-btn').click(function () {
      $('.navbar .menu').toggleClass("active");
      $('.menu-btn i').toggleClass("active");
  });
});

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(2);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
      tf.layers.dense({ units: 100, activation: 'relu' }),
      tf.layers.dense({ units: 2, activation: 'softmax' })
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' }); 
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = await logs.loss.toFixed(5);
        document.getElementById('dummy').innerHTML = loss + "<br> When magic number stops changing click below button"
      }
    }
  });
}

function input(id){
  document.getElementById(id).innerHTML = "<span style='color:#FF0000'>Hold to record upto 50 samples</span>"
}

function timeoutClear() {
  clearTimeout(timer);
}

function handleButton1(id){
    var btn1 = document.getElementById(id);
    btn1.addEventListener('mouseup', timeoutClear);
    btn1.addEventListener('mouseleave', timeoutClear);
}

function handleButton(elem) {
  var btn = document.getElementById(elem.id);
  btn.addEventListener('mousedown', handleButton1(elem.id))
  switch (elem.id) {
    case "0":
      sample1++;
      timer = setTimeout(function() {
        handleButton(elem);
      }, 200);
      document.getElementById("sample-count1").innerText = " Samples: " + sample1;
      break;
    case "1":
      sample2++;
      document.getElementById("sample-count2").innerText = " Samples:" + sample2;
      timer = setTimeout(function() {
        handleButton(elem);
      }, 200);
      break;
  }
  label = parseInt(elem.id);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);

}


async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    var a = document.getElementById('sample1').value;
    var b = document.getElementById('sample2').value;
    const classId = (await predictedClass.data())[0];
    switch (classId) {
      case 0:
        stopPredicting();
        responsiveVoice.speak(a);
        break;    
      case 1:
        stopPredicting();
        responsiveVoice.speak(b);
        break;
    }

    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining() {
  train();
}

function startPredicting() {
  isPredicting = true;
  predict();
}

function stopPredicting() {
  isPredicting = false;
  predict();
}

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));
}

init();

$(document).ready(function(){
  $(this).on("click",".add",function(){
    var newfield = '<div><input type="text" id="sample3" class="inputvalue" placeholder="Name of Gesture"><button class="go" onclick="input(\'demo3\');" id="counting3"> &#8594</button>  <button type="button" id=idCount onclick="handleButton(this);" class="hold">Hold</button><div class="change_count" id="sample-count3"><span id="change3"></span> Samples: <span id=count>0</span></div><p id="demo3"></p>';
    $(".container").append(newfield);
  });
});

var idCount = 0;
$('.hold').each(function() {
   $(this).attr('id', idCount);
   idCount++;
});