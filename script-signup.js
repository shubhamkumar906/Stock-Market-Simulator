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