<template>
  <div>
    <div class="row d-flex">
      <Canvas id="canvas-top" ref="canvas"
              :can_predict="can_predict"
              v-on:predict="predict"></Canvas>

      <div class="col-lg-6 col-m-12 mb-4">
        <div id="output-top" class="card bg-default">
          <div class="card-header font-weight-bold">Output</div>
          <div class="card-body output">
            <div id="not-started" class="current-state">
              <div class="centerer">
                <img class="mascot" src="../assets/dog.png">
                <div class="mascot_do_draw">
                  <div class="speech-bubble">
                    Hi!<br>
                    Draw a single digit from 0-9 in the previous input box, and I'll do my best
                    to guess what you drew!
                  </div>
                </div>
              </div>
            </div>
            <div id="thinking" class="current-state">
              <div class="centerer">
                <img class="mascot" src="../assets/dog_think.png">
                <div class="speech-bubble">
                  Im thinking!
                </div>
              </div>
              <div class="bubblingG">
                <span id="bubblingG_1"></span>
                <span id="bubblingG_2"></span>
                <span id="bubblingG_3"></span>
              </div>
            </div>
            <div id="predictions" class="current-state">
              <div class="centerer">
                <img class="mascot" src="../assets/kitty_predictions.png">
                <div class="speech-bubble">
                  <h2>This is my prediction</h2>
                  <span>{{suggested_digit}}</span>
                  Was I correct?
                </div>
              </div>
              <div class="button-container">
                <button @click="restart" class="btn btn-default">Try Again</button>
                <button @click="not_correct" class="btn btn-danger">Not Correct</button>
                <button @click="correct" class="btn btn-success">Correct</button>
              </div>
            </div>

            <div id="incorrect" class="current-state">
              <div class="centerer">
                <img class="mascot" src="../assets/kitty_sad.png">
                <div class="speech-bubble">
                  Can you tell me what digit you drew?
                </div>
              </div>
              <div class="button-container">
                <div class="btn-group" role="group" aria-label="Basic example">
                  <button v-for="i in 10" @click="train(i-1)" type="button" class="btn btn-secondary">{{i-1}}</button>
                  <button @click="restart" type="button" class="btn btn-danger">No</button>
                </div>
              </div>
            </div>

            <div id="improve" class="current-state">
              <div class="centerer">
                <img class="mascot" src="../assets/kitty_train.png">
                <div class="speech-bubble">
                  Thanks! I will use that information to improve my future predictions!
                </div>
              </div>
              <div class="button-container">
                <button @click="restart" class="btn btn-success">Go again</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
  import axios from 'axios'
  import Canvas from './Canvas'

  export default {
    name: "Home",
    data() {
      return {
        can_predict: true,
        // Output Stages are set in the mounted section.
        output_stages: {},
        suggested_digit: 0
      }
    },
    components: {
      Canvas
    },
    methods: {
      clear: function () {
        this.can_predict = true;
        // I pass true as a force, since the can_predict prop does not
        // seem to update in canvas in time
        // This would also work, but Vue warns not to mutate props directly.
        // this.$refs.canvas.can_predict = true;
        this.$refs.canvas.clear(true);
      },
      restart: function () {
        this.clear();
        this.set_output_stage(this.output_stages.start)
        this.scroll_to('input')
      },
      predict: function () {
        if (this.can_predict) {
          this.can_predict = false;
          this.set_output_stage(this.output_stages.thinking)
          this.scroll_to('output')
          const path = 'http://digits.simonolsen.no/api/predict';
          let canvas_data_url = this.$refs.canvas.getDataUrl();
          axios.post(path, {
            data_url_base_64: canvas_data_url
          })
            .then(response => {
              this.suggested_digit = response.data.prediction
              this.set_output_stage(this.output_stages.prediction)
              this.scroll_to('output')
            })
            .catch(error => {
              console.log(error)
            })
        }
      },
      scroll_to: function (box) {
        if (box === 'input') {
          document.getElementById('canvas-top').scrollIntoView({
            behavior: 'smooth'
          });
        } else if (box === 'output') {
          document.getElementById('output-top').scrollIntoView({
            behavior: 'smooth'
          });
        }
      },
      correct: function () {
        this.set_output_stage(this.output_stages.training)
      },
      not_correct: function () {
        this.set_output_stage(this.output_stages.incorrect)
      },
      train: function (actual_value) {
        this.set_output_stage(this.output_stages.training)
      },
      set_output_stage: function (stage) {
        this.hide_all();
        if (stage === this.output_stages.start) {
          this.output_stages.start.style.maxHeight = "390px";
        }
        else if (stage === this.output_stages.thinking) {
          this.output_stages.thinking.style.maxHeight = "500px";
        }
        else if (stage === this.output_stages.prediction) {
          this.output_stages.prediction.style.maxHeight = "600px";
        }
        else if (stage === this.output_stages.incorrect) {
          this.output_stages.incorrect.style.maxHeight = "550px";
        }
        else if (stage === this.output_stages.training) {
          this.output_stages.training.style.maxHeight = "390px";
        }
      },
      hide_all: async function () {
        this.output_stages.start.style.maxHeight = 0;
        this.output_stages.thinking.style.maxHeight = 0;
        this.output_stages.prediction.style.maxHeight = 0;
        this.output_stages.incorrect.style.maxHeight = 0;
        this.output_stages.training.style.maxHeight = 0;
      }
    },
    mounted() {
      this.$store.commit('change_component', {name: this.$options.name});
      this.output_stages = {
        start: document.getElementById('not-started'),
        thinking: document.getElementById('thinking'),
        prediction: document.getElementById('predictions'),
        incorrect: document.getElementById('incorrect'),
        training: document.getElementById('improve')
      }
    }
  }
</script>

<style lang="scss" scoped>
  @import "../assets/sass/sass_home";
</style>
