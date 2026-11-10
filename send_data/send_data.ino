#define SERVO_CAMERA_PIN 2
#define SERVO_DRIVE_PIN 4
#define ESC_MOTOR_PIN 5
#define NE_PIN 25

#define max_left 120
#define max_right 60
#define center 90

#define chanel_servo_drive 0
#define chanel_servo_camera 1
#define chanel_esc_motor 2

#define F_servo 50
#define F_esc 50

#define ESOLUTION 16

const uint16_t min_pulse_servo = 1636; //0 độ
const uint32_t max_pulse_servo = 6826; //180 độ

uint8_t dem, step_drive;
 
uint8_t angle_drive = center;
uint8_t angle_camera = 45;

uint32_t PWM_servo_camera, PWM_servo_drive;
uint16_t lim_motor = 4720, PWM_ESC_motor = 4000;

String inputString = "";

uint8_t idxA, idxE;
uint8_t angle_drive_get;
uint8_t lim_angle_drive;
float tghientai, tgtruoc, delta;

bool run_motor = false;

bool f = false;
uint32_t tg_ht_ne, tg_tr_ne;
uint16_t chuky = 1, dt, ob;
int van_toc, v_set = 1000;

void IRAM_ATTR NE() {
  
    tg_ht_ne = micros();
    dt = tg_ht_ne - tg_tr_ne;

    if (dt < 200) return;

    if (chuky != 0 && dt < chuky * 0.3) return;
    
    chuky = dt;
    tg_tr_ne = tg_ht_ne;
     
}


void setup() {
  Serial.begin(115200);
  Serial.println("Arduino ready...");
  Serial.begin(115200);

  pinMode(NE_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(NE_PIN),  NE, FALLING);

  ledcSetup(chanel_servo_camera, F_servo, ESOLUTION); 
  ledcAttachPin(SERVO_CAMERA_PIN, chanel_servo_camera); 

  ledcSetup(chanel_servo_drive, F_servo, ESOLUTION);
  ledcAttachPin(SERVO_DRIVE_PIN, chanel_servo_drive); 

  ledcSetup(chanel_esc_motor, F_esc, ESOLUTION);
  ledcAttachPin(ESC_MOTOR_PIN, chanel_esc_motor ); 
  
  ledcWrite(chanel_esc_motor, PWM_ESC_motor);
 
  angle_drive = center;
  PWM_servo_camera = map(angle_camera, 0, 180, min_pulse_servo, max_pulse_servo);
  PWM_servo_drive = map(angle_drive, 0, 180, min_pulse_servo, max_pulse_servo);
  ledcWrite(chanel_servo_camera, PWM_servo_camera);
  ledcWrite(chanel_servo_drive, PWM_servo_drive);
  
}

void loop() {
  

  if (micros() - tg_ht_ne >= 30000) van_toc = 0;
  else {
    van_toc = 32 * (314 * 2 * 13) * 1000 / (chuky * 41);
  }
  
  while (Serial.available()) {
    char c = (char)Serial.read();
    inputString += c;

    if (c == '\n') {
    
      idxA = inputString.indexOf('a');
      idxE = inputString.indexOf('b');
  

      if (idxA != -1 && idxE != -1) {
        angle_drive_get = inputString.substring(idxA + 1, idxE).toInt();
        ob = inputString.substring(idxE + 1).toInt();
        
      }

      inputString = "";
    }
  }


  if (ob == 2) {
    lim_motor = 4710;
    f = false;
  }
  else if (ob == 0) {
    lim_motor = 4710;
    f = false;
  }
  else if (ob == 1 && f == false) {
    f = true;
      if (f == true) {
      lim_motor = 4520;
      PWM_ESC_motor = 4300;
    }angle_drive = 90;
  }

  
  
  tghientai = micros();
  if (tghientai - tgtruoc >= 50000) {
   
    
    if (lim_motor > PWM_ESC_motor) { 
      PWM_ESC_motor += 2;
    }
    else if (lim_motor < PWM_ESC_motor) {
      PWM_ESC_motor -= 2;
    }
    tgtruoc = tghientai;
  }
  
  if (angle_drive > angle_drive_get){
    angle_drive -= 1;
    if (angle_drive < angle_drive_get){
       angle_drive = angle_drive_get;
    }
  }

   else if (angle_drive < angle_drive_get){
    angle_drive += 1;
    if (angle_drive > angle_drive_get){
       angle_drive = angle_drive_get;
    }
   }
  
  PWM_servo_camera = map(angle_camera, 0, 180, min_pulse_servo, max_pulse_servo);
  PWM_servo_drive = map(angle_drive, 0, 180, min_pulse_servo, max_pulse_servo);
  ledcWrite(chanel_servo_camera, PWM_servo_camera);
  ledcWrite(chanel_servo_drive, PWM_servo_drive);
  Serial.print("PWM_ESC_motor: ");
  Serial.println(PWM_ESC_motor);
  if (van_toc != 0) {
    
  }
  //Serial.print("chuky: ");
  //Serial.println(chuky);
  //Serial.print("van_toc: ");
  //Serial.println(van_toc);
  Serial.print("ob: ");
  Serial.println(ob);
  ledcWrite(chanel_esc_motor, PWM_ESC_motor);
  delay(5);
}
