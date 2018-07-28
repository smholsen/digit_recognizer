<template>
  <div class="drawer">
      <ul>
        <li v-bind:class="{ active: current === 'Home'}">
          <a href="/">
            <i class="fas fa-home"></i>
            Home
          </a>
        </li>
        <li v-bind:class="{ active: current === 'About'}">
          <a href="/about">
            <i class="fas fa-info-circle"></i>
            About
          </a>
        </li>
      </ul>
    </div>
</template>

<script>
  export default {
    name: "Menu",
    data() {
      return {
        start_width: window.innerWidth
      }
    },
    methods: {
    },
    computed: {
      current() {
        return this.$store.state.component.name
      }
    },
    mounted: function () {
      console.log(this.$store.component)
    }
  }
</script>

<style lang="scss" scoped>
  .drawer {
    overflow: hidden;
  }
  @media screen and (min-width: 1599px){
    .drawer {
      width: 250px;
      height: 100%;
      position: fixed;
      background-color: #2f2f2f;
      padding-top: 10px;
      ul {
        list-style-type: none;
        padding: 0;

        li {
          display: block;
          text-indent: 14px;
          line-height: 42px;
          font-size: 16px;
          cursor: pointer;
          color: white;
          white-space: nowrap;
          &:hover {
            background-color: #4f4f4f;
          }
          &.active {
            background-color: #4f4f4f;
          }
          i {
            font-size: 18px;
            margin-right: 20px;
          }
          a {
            width: 100%;
            height: 100%;
            display: block;
            text-decoration: none;
            color: white;
          }
        }
      }
    }
  }
  @media screen and (max-width: 1600px){
    .drawer {
      width: 100%;
      max-height: 140px;
      position: relative;
      background-color: white;
      padding-top: 10px;
      ul {
        list-style-type: none;
        padding: 0;
        border-bottom: 1px solid gray;
        border-top: 1px solid gray;
        li {
          display: block;
          text-indent: 14px;
          line-height: 42px;
          font-size: 16px;
          cursor: pointer;
          color: gray;
          white-space: nowrap;
          &:hover {
            background-color: lightgrey;
          }
          &.active {
            background-color: lightgrey;
          }
          i {
            font-size: 18px;
            margin-right: 20px;
          }
          a {
            width: 100%;
            height: 100%;
            display: block;
            text-decoration: none;
            color: black;
          }
        }
      }
    }
  }
</style>
