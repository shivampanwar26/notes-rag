import axios from "axios";

// One shared axios instance with the backend's base URL. Every component
// imports this instead of hard-coding "http://localhost:8000" repeatedly.
const api = axios.create({
  baseURL: "http://localhost:8000",
});

export default api;
