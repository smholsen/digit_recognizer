<template>
    <div>
      <p>
        About
        Dog Icon from <a target="_blank" href="http://www.freepik.com/">Freepik</a>
        from <a target="_blank" href="www.flaticon.com">www.flaticon.com</a>
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

<style scoped>

</style>
