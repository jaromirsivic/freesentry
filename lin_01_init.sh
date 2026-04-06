# installation
echo "update and upgrade the system"
sudo apt update
sudo apt full-upgrade

echo "create freesentry directory"
cd ~
mkdir freesentry

echo "install uv"
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "install samba"
sudo apt install -y samba samba-common-bin
echo -e "[global]\nworkgroup = WORKGROUP\nserver string = freesentry\nnetbios name = freesentry\nsecurity = user\n# Protocol settings for modern Windows 10/11 compatibility\nserver min protocol = SMB3_11\nclient min protocol = SMB3\nmap to guest = bad user\n\n[freesentry]\ncomment = freesentry\npath = /home/freesentry/freesentry\nbrowsable = yes\nwritable = yes\nonly guest = no\nvalid users = freesentry\nforce create mode = 0660\nforce directory mode = 0770" | sudo tee /etc/samba/smb.conf
sudo smbpasswd -a freesentry
sudo systemctl restart smbd
sudo systemctl restart nmbd
sudo chown -R freesentry:freesentry /home/freesentry/freesentry
sudo chmod -R 755 /home/freesentry/freesentry

echo "allow port 80 to be used by freesentry server"
echo -e "net.ipv4.ip_unprivileged_port_start=0" | sudo tee -a /etc/sysctl.d/98-rpi.conf

echo "enable hardware pwm"
echo -e "# pwm setup for rpi4\ndtoverlay=pwm-2chan,pin=12,func=4,pin2=18,func=2\ndtoverlay=pwm-2chan,pin=13,func=4,pin2=19,func=2" | sudo tee -a /boot/firmware/config.txt
sudo reboot

# restart putty session
# copy package folder to freesentry directory
cd ~
cd freesentry

echo "init python venv"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
uv init
uv add fastapi --extra standard
uv add uvicorn --extra standard
uv add numpy
uv add opencv-python
uv pip install "opencv-python-headless"
uv add ultralytics
uv add debugpy
uv pip install python-periphery
sudo apt install -y python3-picamera2
sudo apt install python3-lgpio

# create start script
echo '#!/bin/bash
source /home/freesentry/freesentry/.venv/bin/activate
uvicorn server.main:app --app-dir /home/freesentry/freesentry --host 0.0.0.0 --port 80' | sudo tee /home/freesentry/freesentry/lin_01_start.sh
sudo chmod +x /home/freesentry/freesentry/lin_01_start.sh
sudo chown freesentry:freesentry /home/freesentry/freesentry/lin_01_start.sh

echo "add freesentry to crontab"
sudo su
crontab -e
@reboot /home/freesentry/freesentry/rpi_01_start.sh &
#{ sudo crontab -l -u root; echo '@reboot /home/freesentry/freesentry/rpi_01_start.sh &'; } | sudo crontab -u root -
sudo reboot

# start freesentry server
echo "start freesentry server"
cd /home/freesentry/freesentry
source .venv/bin/activate
uvicorn server.main:app --app-dir /home/freesentry/freesentry --host 0.0.0.0 --port 80

# create image of sd card:
#wget https://raw.githubusercontent.com/Drewsif/PiShrink/master/pishrink.sh
#sudo chmod +x pishrink.sh
#sudo mv pishrink.sh /usr/local/bin
sudo apt-get clean
df -ah --total
# insert empty usb flash
lsblk
sudo mkdir /dev/usb
sudo mount /dev/sda1 /dev/usb
# count should be the size of MB returned by df -ah + 1000
sudo dd if=/dev/mmcblk0 of=/dev/usb/freesentry.img bs=1M count=7000
sudo umount /dev/usb


# found commands

echo "update and upgrade the system"
sudo apt update
sudo apt full-upgrade

echo "install uv"
cd ~
curl -LsSf https://astral.sh/uv/install.sh | sh
exit

# restart putty session
cd ~
mkdir freesentry
cd freesentry

python3 -m venv --system-site-packages .venv

# clone the repository and initialize uv
git clone https://github.com/jaromirsivic/freesentry.git .
uv init
#uv venv --python 3.14
#rem sudo chmod 777 ./.venv/bin/activate

source .venv/bin/activate
uv add fastapi --extra standard
uv add uvicorn --extra standard
uv add numpy
uv add opencv-python
uv pip install "opencv-python-headless"
uv add ultralytics
uv add debugpy
uv add picamera2
sudo apt install -y python3-picamera2
sudo apt install python3-lgpio
uv pip install python-periphery
# hardware pwm


# installation of nvidia gpu support for yolo models
sudo apt install nvidia-cuda-toolkit
sudo apt install nvidia-cuda-toolkit-dev
sudo apt install nvidia-cuda-toolkit-doc
sudo apt install nvidia-cuda-toolkit-samples
sudo apt install nvidia-cuda-toolkit-examples
sudo apt install nvidia-cuda-toolkit-tests
sudo apt install nvidia-cuda-toolkit-tools
sudo apt install nvidia-cuda-toolkit-utils
sudo apt install nvidia-cuda-toolkit-libs
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


sudo sysctl -w net.ipv4.ip_unprivileged_port_start=0
sudo nano /etc/sysctl.d/98-rpi.conf
echo -e "net.ipv4.ip_unprivileged_port_start=0" | sudo tee -a /etc/sysctl.d/98-rpi.conf
INSERT NEW LINE WITH net.ipv4.ip_unprivileged_port_start=0

sudo apt install -y samba samba-common-bin
echo -e "[global]\nworkgroup = WORKGROUP\nserver string = freesentry\nnetbios name = freesentry\nsecurity = user\n# Protocol settings for modern Windows 10/11 compatibility\nserver min protocol = SMB3_11\nclient min protocol = SMB3\nmap to guest = bad user\n\n[freesentry]\ncomment = freesentry\npath = /home/freesentry/freesentry\nbrowsable = yes\nwritable = yes\nonly guest = no\nvalid users = freesentry\nforce create mode = 0660\nforce directory mode = 0770" | sudo tee /etc/samba/smb.conf

# enable hardware pwm on raspberry pi 5
# disable audio in /boot/firmware/config.txt
dtparam=audio=off
echo -e "# pwm setup for rpi5\ndtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func=4\ngpio=18=a3\ngpio=19=a3" | sudo tee -a /boot/firmware/config.txt
sudo reboot

# enable hardware pwm on raspberry pi 4
# disable audio in /boot/firmware/config.txt
dtparam=audio=off
# Enable PWM0 on GPIO 12 and 18
dtoverlay=pwm-2chan,pin=12,func=4,pin2=18,func=2
# Enable PWM1 on GPIO 13 and 19
dtoverlay=pwm-2chan,pin=13,func=4,pin2=19,func=2
echo -e "# pwm setup for rpi4\ndtoverlay=pwm-2chan,pin=12,func=4,pin2=18,func=2\ndtoverlay=pwm-2chan,pin=13,func=4,pin2=19,func=2" | sudo tee -a /boot/firmware/config.txt
sudo reboot

# sudo nano /etc/samba/smb.conf
# [global]
#    workgroup = WORKGROUP
#    server string = freesentry
#    netbios name = freesentry
#    security = user
#    # Protocol settings for modern Windows 10/11 compatibility
#    server min protocol = SMB3_11
#    client min protocol = SMB3
#    map to guest = bad user

# [freesentry]
#    comment = freesentry
#    path = /home/freesentry/freesentry
#    browsable = yes
#    writable = yes
#    only guest = no
#    valid users = freesentry
#    force create mode = 0660
#    force directory mode = 0770

sudo smbpasswd -a freesentry
sudo systemctl restart smbd
sudo systemctl restart nmbd

sudo chown -R freesentry:freesentry /home/freesentry/freesentry
sudo chmod -R 755 /home/freesentry/freesentry
cd /home/freesentry/freesentry
# copy project except of .git and .venv
uv sync

# system wide installation
#sudo apt install python3-uvicorn
#sudo apt install python3-fastapi
#sudo apt install python3-pigpio

# enable wifi access point wpa psk
sudo nmcli device wifi hotspot ssid default password freesentry
sudo nmcli device wifi hotspot ifname wlan0 ssid "default" password "freesentry"
# wpa3
sudo nmcli connection add type wifi ifname wlan0 con-name "WPA3-Hotspot" autoconnect yes ssid "MyWPA3Network" -- 802-11-wireless.mode ap 802-11-wireless.band bg ipv4.method shared wifi-sec.key-mgmt sae wifi-sec.psk "YourStrongPassword"
# disable wifi access point
sudo nmcli device wifi hotspot off
# scan for wifi networks and save the results to a file, it allows scanning for at most 10 seconds
sudo nmcli device wifi list > wifi_networks.txt
# connect to a wifi network
sudo nmcli device wifi connect "SSID" password "PASSWORD"
# disconnect from a wifi network
sudo nmcli device wifi disconnect
# show the current wifi connection
sudo nmcli device wifi show
# show the wifi networks
sudo nmcli device wifi list
# wifi restart
sudo nmcli radio wifi off && sudo nmcli radio wifi on
# connect to wifi using wpa psk
sudo nmcli connection add type wifi ifname wlan0 con-name "D-Link" ssid "D-Link" -- wifi-sec.key-mgmt wpa-psk wifi-sec.psk "ChlupateLejno"
# connect using no password
sudo nmcli connection add type wifi ifname wlan0 con-name "FreeWifi" ssid "FreeWifi"
# connect using wpa3 personal SAE
sudo nmcli connection add type wifi ifname wlan0 con-name "ModernNet" ssid "ModernNet" -- wifi-sec.key-mgmt sae wifi-sec.psk "YourPassword"
# connect using wpa/wpa2 (802.1X)
# sudo nmcli connection add type wifi ifname wlan0 con-name "OfficeNet" ssid "OfficeNet" -- wifi-sec.key-mgmt wpa-eap 802-1x.eap peap 802-1x.identity "your_username" 802-1x.password "your_password" 802-1x.phase2-auth mschapv2
# connect using WEP
sudo nmcli connection add type wifi ifname wlan0 con-name "OldNet" ssid "OldNet" -- wifi-sec.key-mgmt none wifi-sec.wep-key0 "YourWepKey" wifi-sec.auth-alg open

# debug mode
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn server.main:app --host 0.0.0.0 --port 80
# production mode
uvicorn server.main:app --host 0.0.0.0 --port 80

https://www.jeffgeerling.com/blog/2023/nmcli-wifi-on-raspberry-pi-os-12-bookworm/

# pigpio daemon
cd /home
sudo apt install python-setuptools python3-setuptools
wget https://github.com/joan2937/pigpio/archive/master.zip
unzip master.zip
cd pigpio-master
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
make
sudo make install

# sleep and wakeup
https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#real-time-clock-rtc



rem git init
rem IF NOT EXIST .venv\Scripts\activate (uv venv)
call .venv\Scripts\activate
rem uv pip install -U ultralytics
rem uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add fastapi --extra standard
uv add uvicorn --extra standard
uv add gpiozero pigpio
pause

# start on startup
# add folowing line into corntab. sudo crontab -e
@reboot /home/freesentry/freesentry/lin_01_init.sh &