<?php 
$img = $_POST['input'];
$img = str_replace('data:image/png;base64,', '', $img);
$img = str_replace(' ', '+', $img);
// echo $img;
$data = base64_decode($img);
$file = "inp.png";
// $success = file_put_contents($file, $data);
?>