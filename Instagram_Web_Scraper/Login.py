class Login:
    ## this gets the website. Input into class username and then password


    def __init__(self, username, password):
        self.username = username
        self.password = password


    def insta_page(self):

        driver.get(f"https://www.instagram.com/{self.username}/?hl=en")

        time.sleep(random.uniform(1.5, 2))

    def password_insta_logger_other(self):

        id_box = driver.find_element_by_name('username')
        # Select password box
        pass_box = driver.find_element_by_name('password')
        # Send id information ## enter your username and password in
        id_box.send_keys(self.username)
        # Send password
        pass_box.send_keys(self.password)
        # Find login button
        login_button = driver.find_element_by_xpath('//*[@id="loginForm"]/div/div[3]/button/div')
        # Click login
        login_button.click()
        ##clear notification button
        time.sleep(random.uniform(1, 2))



    def password_insta_logger(self):


        ##Select the log in page
        log_page = driver.find_element_by_xpath("// *[ @ id = \"react-root\"] / section / nav / div[2] / div / div / div[3] / div / span / a[1]")
        log_page.click()
        time.sleep(random.uniform(2, 3))
        ##log in
        id_box = driver.find_element_by_name('username')
        # Select password box
        pass_box = driver.find_element_by_name('password')
        # Send id information ## enter your username and password in
        id_box.send_keys(self.username)
        # Send password
        pass_box.send_keys(self.password)
        # Find login button
        login_button = driver.find_element_by_xpath('//*[@id="loginForm"]/div/div[3]/button/div')
        # Click login
        login_button.click()
        ##clear notification button
        time.sleep(random.uniform(2, 3))
        profile_notification_off_login = driver.find_element_by_xpath("//*[@id=\"react-root\"]/section/main/div/div/div/div")
        profile_notification_off_login.click()
