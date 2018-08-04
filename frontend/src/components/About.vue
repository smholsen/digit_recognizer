<template>
    <div class="ab_wrapper">
      <h1>About</h1>
      <div class="btn-group">
        <a href="#general">
          <button class="btn btn-primary">
          General
          </button>
          </a>
        <a href="#setup">
          <button class="btn btn-primary">
            Setup
          </button>
        </a>
        <a href="#model">
          <button class="btn btn-primary">
            Model
          </button>
        </a>
        <a href="#model_webapp">
          <button class="btn btn-primary">
            Challenges in Web App Combination
          </button>
        </a>
        <a href="#make_your_own">
          <button class="btn btn-primary">
            Guide to Set Up
          </button>
        </a>
      </div>

      <div class="alert alert-primary center" role="alert">
        <p>
          This is still a Work in Progress.<br>
          <br>
          The model is now able to train on your drawings! You can now help the cat get smarter!
          <br>
          ToDos:
          <ul>
            <li>Improve Documentation & Write Guide and Readme</li>
          </ul>
          All of the code for the project can be found on https://github.com/smholsen/digit_recognizer
        </p>
      </div>

      <div id="general">
        <h2>General</h2>
        I started working on this project with a desire to learn how machine learning models could be applied to some problem. The idea spawned from the <a href="https://www.kaggle.com/c/digit-recognizer" target="_blank@">MNIST Digit Recognizer challenge on Kaggle</a>, where users are tasked with designing a classifier model that should be able to classify images of hand-drawn digits with a degree of accuracy as high as possible. The process of designing the final model was highly educational, and I've written more about this in the <a href="#model">section about the model</a>.
        <br>
        <br>
        After designing and training a model that performed acceptably on the MNIST dataset, I started the work on a web application that could use the model in an interactive way. Since I already had a Nginx web server running on my DigitalOcean droplet, I found a Flask backend through an <a href="http://flask.pocoo.org/docs/1.0/deploying/uwsgi/">uWSGI server</a> to be the best alternative. For the frontend, I quickly decided to go with Vue.js for a simple and clear separation of components.
        <br>
        <br>
        Since I had no previous experience with HTML5 Canvas, I spent quite some time figuring out how to best allow the user to draw their digits, but in the end, I think it turned out quite nice. It was some hassle to make it work nicely with touch in addition to standard mouse movements, but I found this to be quite an important feature. From my personal experience, I actually found the digits that were drawn by touch to have a higher probability of being correctly classified by the network that was solely trained on the MNIST dataset. I guess this makes sense, as the digits from the dataset were all hand-drawn, and they probably tend to look somewhat different than digits drawn with a mouse or touchpad.
      </div>
      <div id="setup">
        <h2>Setup</h2>
        Below follows a brief introduction to the technologies and services required to run the application. For a more complete overview, please see the section on <a href="#make_your_own">how you can set up your own version</a>.
        <br>
        <br>
        <ul>
          <li>Nginx</li>
          <li>uWSGI</li>
          <li>Flask</li>
          <li>Python3</li>
          <li>Vue.js</li>
        </ul>
      </div>
      <div id="model">
        <h2>Model</h2>
        The model used in the prediction of digits is a multi-layered convolutional neural network made using TensorFlow.
        Since I used the MNIST dataset as a starting point for the project, I decided to keep the input size of the drawn images similar to the images from the MNIST dataset (28x28).
        Thus, the input layer of the network was set to be a set of 784, (28x28), input neurons, each containing the pixel value of its respective pixel from the image. You can imagine taking each pixel row of the image and placing them side by side. This would be the input layer.
        <br>
        <br>
        The next layer in our network is the first convolutional layer.
        If convolution is an unknown term for you, then I highly recommend watching <a href="https://www.youtube.com/watch?v=FmpDIaiMIeA" target="_blank">this 25-minute youtube video</a> by <a href="https://www.youtube.com/channel/UCsBKTrp45lTfHa_p49I2AEQ" target="_blank">Brandon Rohrer</a> on how CNN's work. It's really good!
        Now, in the model's first convolutional layer, we look at each 5x5 patch (each sub-image in the complete image that is of size 5x5 pixels) and evaluate 32 different features. Features, in this case, are patterns in the image, for example, a slanting line or a straight line. These 32 features are defined during the models learning-process. The features are used to filter the image, obtaining one filtered image for each feature.
        In the pooling, we use strides of 2x2 pixels to traverse each of the filtered images from the filtered image stack. During the pooling, we keep the max value from the stride and thus shrink the image.
        After the first convolutional layer is done, we are left with images of size 14x14 pixels.
        <br>
        <br>
        The second convolutional layer repeats the steps from the first but considers 64 features for each patch. The images have now been reduced to a size of 7x7 pixels.
        <br>
        <br>
        In the next layer, we connect each of the pixels from each of the 7x7 images to neurons in the next layer. The number of neurons in the next layer is defined by us, and we set this to 1024.
        <br>
        <br>
        In the final layer, each of the 1024 neurons gives a confidence score to each of the 10 possible output classes, (digits 0 through 9). For this, we simply use matrix multiplication, and higher values yield higher scores. The class with the highest score wins and is our prediction.
        <br>
        <br>
        During the initial training of the model, I applied <a href="https://en.wikipedia.org/wiki/Dropout_(neural_networks)" target="_blank">Dropout</a> in an attempt to avoid overfitting. This seemed to improve the model's robustness.
      </div>

      <div id="model_webapp">
        <h2>Challenges in Combination of Web App & Model</h2>
        In the previous section, I noted that the images used as input for the model would require a pixel size of 28x28. However, requiring the users to draw digits on such a small canvas would feel restrictive and cumbersome. I, therefore, decided to allow the drawings to be placed on a bigger canvas, and then process the images before feeding them to the network. This caused some issues. One issue we can identify relatively quickly is that users tend to draw digits in different sizes, and all of the data from the original dataset were drawn in such a way so that the digit was almost always about 1 or 2 pixels away from the closest edges. For example, a 1 would be close to both the top and bottom edge, while a 5 would be close to all.
        To combat this effect, I used the Image module from the Python library Pillow to detect the actual size of the drawn digit. After detecting the digit I could cut out only the digit from the original canvas, resize it, and paste it into a 28x28 pixel image that could be used as input for the network. This allowed the input to more closely resemble the data used in training and testing the model.
      </div>

      <div id="make_your_own">
        <h2>Guide to Set Up Project</h2>
        This section has not yet been written, but will contain a guide on how you can set this entire project up on your own and make it better!
      </div>

      <h2>
        Misc
      </h2>
      <p>
        Icons from <a target="_blank" href="https://www.freepik.com/">Freepik</a>
        from <a target="_blank" href="https://www.flaticon.com">www.flaticon.com</a>
      </p>
    </div>
</template>

<script>
    export default {
        name: "About",

      mounted() {
        this.$store.commit('change_component', {name: this.$options.name})
      }

    }
</script>

<style lang="scss" scoped>
  @import "../assets/sass/sass_about.scss";
</style>
