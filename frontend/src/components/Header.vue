<template>
  <div>
    <nav class="header">
      <h1>Digit Recognizer</h1>
      <div class="nav-btn" @click="toggle_nav_drawer">
        <i class="fas fa-bars"></i>
      </div>
    </nav>
  </div>
</template>

<script>
  export default {
    name: "Header",
    methods: {
      toggle_nav_drawer: function () {
        this.$emit('toggle_nav')
      }
    }
  }
</script>

<style lang="scss" scoped>
  .header {
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 10px;
    display: flex;
    position: relative;
    justify-content: center;
    h1 {
      color: rgba(0, 0, 0, 0.9);
      margin-right: 1rem;
      font-size: 1.25rem;
      line-height: inherit;
      white-space: nowrap;
    }
    .nav-btn {
      display: inline-block;
      position: absolute;
      text-align: center;
      left: 10px;
      i {
        cursor: pointer;
        border: 1px solid rgba(0, 0, 0, .1);
        border-radius: 5px;
        text-decoration: none;
        margin: 0;
        line-height: 25px;
        padding: 5px;
        width: 40px;
        font-size: 20px;
        color: gray;
      }
    }
  }
</style>
