##used for accounts that are not yours
class Public_account:
    def __init__(self, username):
        self.username = username

    def insta_page2(self):
        print(f"https://www.instagram.com/{self.username}/?hl=en")
        driver.get(f"https://www.instagram.com/{self.username}/?hl=en")
        time.sleep(random.uniform(.75, 1.5))
