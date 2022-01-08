class insta_navigator:
  @staticmethod
  def not_now():
    profile_notification_off_login = driver.find_element_by_xpath("//*[@id=\"react-root\"]/section/main/div/div/div/div")
    profile_notification_off_login.click()
  @staticmethod
  def profile_reacher():
    profile_clicker = driver.find_element_by_xpath("//*[@id=\"react-root\"]/section/main/div/div/div/div")
    profile_clicker.click()

  @staticmethod
  def follower_finder():
    time.sleep(random.uniform(.5, 1))
    #now you are within the profile. next we fill click on followers and collect followers name, for some reason xpath from chrome does not work properly
    followers_button=driver.find_element_by_xpath("//*[@id=\"react-root\"]/section/main/div/header/section/ul/li[2]/a")
    followers_button.click()

  def getUserFollowers(self, max):
    time.sleep(random.uniform(.75, 1))
    ##opens up links of followers
    followersLink = driver.find_element_by_xpath("//*[@id=\"react-root\"]/section/main/div/header/section/ul/li[2]/a")
    followersLink.click()
    time.sleep(1)
    ##selects the list of followers
    FList = driver.find_element_by_css_selector('div[role=\'dialog\'] ul')

    numberOfFollowersInList = len(FList.find_elements_by_css_selector('li'))

    FList.click()
    actionChain = webdriver.ActionChains(driver)
    time.sleep(random.randint(2, 4))
    actionChain.key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
    followingbox = driver.find_element_by_xpath("/html/body/div[4]/div/div/div[2]")
    actionChain.key_down(Keys.SPACE).key_up(Keys.SPACE).perform()



    while (numberOfFollowersInList +1 <max):


        actionChain.key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        numberOfFollowersInList = len(FList.find_elements_by_css_selector('li'))
        time.sleep(0.2)
        followingbox.click()
        actionChain.key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        time.sleep(.5)

    followers = []
    for user in FList.find_elements_by_css_selector('li'):
        userLink = user.find_element_by_css_selector('a').get_attribute('href')

        followers.append(userLink)
        if (len(followers) == max):
            break
    close = driver.find_element_by_xpath("/html/body/div[4]/div/div/div[1]/div/div[2]/button")
    close.click()
    return followers


  def getUserFollowing(self, max):
    time.sleep(.5)
    followingLink = driver.find_element_by_xpath("//*[@id=\"react-root\"]/section/main/div/header/section/ul/li[3]")
    followingLink.click()
    time.sleep(.5)
    FList = driver.find_element_by_css_selector('div[role=\'dialog\'] ul')
    FList.click()
    time.sleep(.5)
    numberOfFollowingInList = len(FList.find_elements_by_css_selector('li'))


    actionChain = webdriver.ActionChains(driver)
    time.sleep(random.randint(2, 4))

    followingbox = driver.find_element_by_xpath("/html/body/div[4]/div/div/div[2]")
    followingbox.click()

    while (numberOfFollowingInList +1 < max):
        actionChain.key_down(Keys.SPACE).key_up(Keys.SPACE).perform()

        numberOfFollowingInList = len(FList.find_elements_by_css_selector('li'))

        time.sleep(0.2)
        followingbox.click()
        actionChain.key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        time.sleep(.5)

    following =[]
    for user in FList.find_elements_by_css_selector('li'):
        userLink = user.find_element_by_css_selector('a').get_attribute('href')

        following.append(userLink)
        if (len(following) == max):
            break
    close = driver.find_element_by_xpath("/html/body/div[4]/div/div/div[1]/div/div[2]/button")
    close.click()
    return following
  def comparer(self):
    navi= insta_navigator()
    ##this colllects the number of followers and following
    no_following_init = driver.find_element_by_xpath("//*[@id=\"react-root\"]/section/main/div/header/section/ul/li[3]/a/span[@class = 'g47SY ']").text

    no_following= int(no_following_init.replace(",", ""))
    no_followers_init= driver.find_element_by_xpath("//*[@id=\"react-root\"]/section/main/div/header/section/ul/li[2]/a/span").get_attribute("title")
    no_followers = int(no_followers_init.replace(",", ""))

    ##this function compares the two arrays and finds the differences
    traitors = np.setdiff1d(navi.getUserFollowing(no_following),navi.getUserFollowers(no_followers))
    ##this makes it so only the username is printed
    b = []
    for s in traitors:
        i1 = s[25:]
        b.append(i1)
    print(b)
