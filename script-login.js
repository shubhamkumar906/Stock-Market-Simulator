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