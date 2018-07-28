<template>
  <div id="app">
    <Header v-on:toggle_nav="toggle_nav()"></Header>
    <Menu v-bind:class="{hide: !menu_toggled}"></Menu>
    <div class="wrapper" v-bind:class="{pad: menu_toggled}">
      <router-view/>
    </div>
  </div>
</template>

<script>
  import Header from './components/Header'
  import Menu from './components/Menu'

  export default {
    name: 'App',
    components: {
      Header,
      Menu
    },
    data() {
      return {
        menu_toggled: true
      }
    },
    methods: {
      toggle_nav: function () {
        this.menu_toggled = !this.menu_toggled;
      }
    },
    mounted() {
      if (window.innerWidth < 900) {
        // this.menu_toggled = false;
      }
    }
  }
</script>

<style lang="scss">
  * {
    transition: .5s all;
  }

  body {
    margin: 0;
    font-family: 'Roboto', sans-serif;
    #app {
      width: 100%;

      .wrapper {
        padding: 40px;

        .card {
          box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
          height: 100%;
        }
      }
    }
  }

  .hide {
    @media screen and (max-width: 1599px) {
      max-height: 0 !important;
    }
    @media screen and (min-width: 1600px) {
      width: 0 !important;
    }
  }

  .pad {
    @media screen and (max-width: 1599px) {

    }
    @media screen and (min-width: 1600px) {
      padding-left: 290px !important;
    }
  }
</style>
