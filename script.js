// contact us
function validate(){
  var name = document.getElementById("name").value;
  var phone = document.getElementById("phone").value;
  var email = document.getElementById("email").value;
  var message = document.getElementById("message").value;
  var error_message = document.getElementById("error_message");

  error_message.style.padding = "10px";

  var text;
  if(name.length < 5){
    text = "Please Enter valid Name";
    error_message.innerHTML = text;
    return false;
  }

  if(isNaN(phone) || phone.length != 10){
    text = "Please Enter valid Phone Number";
    error_message.innerHTML = text;
    return false;
  }
  if(email.indexOf("@") == -1 || email.length < 6){
    text = "Please Enter valid Email";
    error_message.innerHTML = text;
    return false;
  }
  if(message.length <= 10){
    text = "Please Enter More Than 10 Characters";
    error_message.innerHTML = text;
    return false;
  }
  alert("Form Submitted Successfully!");
  return true;
}

// login

var email_login=document.getElementById("text1");
var password_login=document.getElementById("text2");
var email_error=document.getElementById("email_error");
var password_error=document.getElementById("password_error");

function validatelogin()
{
    if(email_login.value.length<10)
    {
      email_error.style.display="block";
      return false;
    }
    if( password_login.value.length<1)
    {
      password_error.style.display="block";
      return false;
    }
    alert("SUCCESSFULLY LOGGED-IN");
    return true;
}

// signup
var username=document.getElementById("username");
var password=document.getElementById("password");
var email=document.getElementById("email");
var contact=document.getElementById("contact");

var username_error_signup=document.getElementById("username_error_signup");
var password_error_signup=document.getElementById("password_error_signup");
var email_error_signup=document.getElementById("email_error_signup");
var phone_error_signup=document.getElementById("phone_error_signup");

function validatesignup()
{
  if(username.value.length<1)
  {
    username_error_signup.style.display="block";
    return false;
  }
    if(password.value.length<1)
  {
    password_error_signup.style.display="block";
    return false;
  }
      if(email.value.length<1)
  {
    email_error_signup.style.display="block";
    return false;
  }
      if(contact.value.length<10)
  {
    phone_error_signup.style.display="block";
    return false;
  }
  alert("Account Created :)");
  return true;
}



