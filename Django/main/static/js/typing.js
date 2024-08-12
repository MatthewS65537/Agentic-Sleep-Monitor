document.addEventListener('DOMContentLoaded', function() {
    const text = "Welcome to the Future of Sleep";
    const typingSpeed = 75; // typing speed in milliseconds
    let index = 0;
  
    function type() {
      if (index < text.length) {
        document.getElementById('typing-effect').innerHTML += text.charAt(index);
        index++;
        setTimeout(type, typingSpeed);
      } else {
        // Optional: Add blinking cursor effect after typing is complete
        setTimeout(() => {
          document.getElementById('typing-effect').innerHTML = '';
          index = 0;
          type();
        }, 1600);
      }
    }
    type();
  });
  