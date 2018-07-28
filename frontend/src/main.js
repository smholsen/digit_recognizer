import Vue from 'vue'
import App from './App'
import router from './router'
import Vuex from "vuex";

Vue.config.productionTip = false

Vue.use(Vuex)

const store = new Vuex.Store({
  state: {
    component: {name:'Home'}
  },
  mutations: {
    change_component (state, component) {
      state.component.name = component.name
    }
  }
})

/* eslint-disable no-new */
new Vue({
  el: '#app',
  store,
  router,
  render: h => h(App)
})
