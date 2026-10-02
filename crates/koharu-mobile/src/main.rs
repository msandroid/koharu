// Desktop entry point for developing the mobile app without a device.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    koharu_mobile_lib::run();
}
