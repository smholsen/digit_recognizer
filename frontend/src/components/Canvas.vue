<template>
  <div class="col-lg-6 col-m-12 mb-4">
    <div class="card bg-default">
      <div class="card-header font-weight-bold">Input</div>
      <div class="card-body input">
        <div class="squiggly">
          <span>Draw your digit here </span> <img class="arrow" src="../assets/arrow.png"/>
        </div>
        <div class="canvas-container">
          <canvas id="drawing-board"
                  v-on:mousedown="mouse_down"
                  v-on:mousemove="mouse_move"
                  v-on:mouseup="mouse_up"
                  v-on:mouseleave="mouse_leave"
          >

          </canvas>
        </div>
        <div class="button-container">
          <button @click="clear(false)" v-bind:class="{disabled: !can_predict}" class="btn btn-secondary">
            Clear
          </button>
          <button @click="predict" v-bind:class="{disabled: !can_predict}" class="btn btn-success">
            Predict
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
  // Normal Canvas
  // http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/
  // For Touch
  // https://developer.mozilla.org/en-US/docs/Web/API/Touch_events
  export default {
    name: "Canvas",
    props: [
      'can_predict'
    ],
    data() {
      return {
        // These are for drawing with mouse
        paint: false,
        clickX: [],
        clickY: [],
        clickDrag: [],
        // These are for touch
        ongoingTouches: [],
        // The following are set in mounted
        canvas: null,
        context: null,
        offsetX: 0,
        offsetY: 0
      }
    },
    methods: {
      predict: function () {
        this.$emit('predict');
      },
      // This methos is used by the Home Component to obtain the image from the canvas
      // To use in its post request to the server.
      getDataUrl: function () {
        return this.canvas.toDataURL();
      },
      clear: function (force = false) {
        if (this.can_predict || force) {
          this.context.clearRect(0, 0, this.context.canvas.width, this.context.canvas.height);
          this.clickX = [];
          this.clickY = [];
          this.clickDrag = [];
        }
      },
      set_offset: function () {
        // Set offset
        // This is done on every draw, as it can change with different
        // resizings in the DOM
        let rect = this.canvas.getBoundingClientRect();
        this.offsetX = rect.left;
        this.offsetY = rect.top;
      },
      mouse_down: function (e) {
        if (this.can_predict) {
          this.set_offset();
          let mouseX = e.clientX - this.offsetX;
          let mouseY = e.clientY - this.offsetY;
          this.paint = true;
          this.addClick(mouseX, mouseY);
          this.redraw();
        }
      },
      mouse_move: function (e) {
        this.set_offset();
        if (this.can_predict) {
          if (this.paint) {
            let mouseX = e.clientX - this.offsetX;
            let mouseY = e.clientY - this.offsetY;
            this.addClick(mouseX, mouseY, true);
            this.redraw();
          }
        }
      },
      mouse_up: function (e) {
        this.paint = false;
      },
      mouse_leave: function (e) {
        this.paint = false;
      },
      addClick: function (x, y, dragging) {
        this.clickX.push(x);
        this.clickY.push(y);
        this.clickDrag.push(dragging);
      },
      redraw: function redraw() {
        let self = this;
        this.context.clearRect(0, 0, this.context.canvas.width, this.context.canvas.height); // Clears the canvas
        this.context.strokeStyle = "#000000";
        this.context.lineJoin = "round";
        this.context.lineWidth = 10;

        for (let i = 0; i < this.clickX.length; i++) {
          self.context.beginPath();
          if (self.clickDrag[i] && i) {
            self.context.moveTo(self.clickX[i - 1], self.clickY[i - 1]);
          } else {
            self.context.moveTo(self.clickX[i] - 1, self.clickY[i]);
          }
          self.context.lineTo(self.clickX[i], self.clickY[i]);
          self.context.closePath();
          self.context.stroke();
        }
      },
      initialize_for_touch: function () {
        this.canvas.addEventListener("touchstart", this.handleStart, false);
        this.canvas.addEventListener("touchend", this.handleEnd, false);
        this.canvas.addEventListener("touchcancel", this.handleCancel, false);
        this.canvas.addEventListener("touchmove", this.handleMove, false);
      },
      // These handlers are for Touch
      handleStart: function (evt) {
        evt.preventDefault();
        let touches = evt.changedTouches;

        for (let i = 0; i < touches.length; i++) {
          this.mouse_down(touches[i])
        }
      },
      handleMove: function (evt) {
        evt.preventDefault();
        let touches = evt.changedTouches;

        for (let i = 0; i < touches.length; i++) {
          this.mouse_move(touches[i])
        }
      },
      handleEnd: function (evt) {
        evt.preventDefault();
        this.mouse_up()
      },
      handleCancel: function (evt) {
        evt.preventDefault();
        this.mouse_up()
      }
    },
    mounted() {
      this.context = document.getElementById('drawing-board').getContext("2d");
      this.canvas = document.getElementById('drawing-board');
      // Set correct width & height
      this.context.canvas.width = 200;
      this.context.canvas.height = 200;
      // Initialize for Touch
      this.initialize_for_touch();
    }
  }
</script>

<style lang="scss" scoped>
  .input {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;

    .squiggly {
      text-align: center;
      font-family: 'Indie Flower', cursive;
      font-size: 30px;
      display: inline;

      .arrow {
        width: 50px;
        height: auto;
        display: inline;
        transform: scaleX(-1);
      }
    }

    .canvas-container {
      display: flex;
      justify-content: center;
      margin-top: 20px;
      margin-bottom: 20px;

      #drawing-board {
        cursor: pointer;
        width: 200px;
        height: 200px;
        box-shadow: 0 0 8px 8px rgba(0, 0, 0, 0.2), 6px 6px 6px 6px rgba(0, 0, 0, 0.19);
      }

    }

    .button-container {
      display: flex;
      justify-content: flex-end;

      .btn {
        margin-left: 20px;
        box-shadow: none;
      }

    }
  }
</style>
