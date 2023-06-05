### 1.创建第一个Kotlin应用程序
## （1）创建一个新的工程
打开Android Studio，选择Projects>New Project，然后选择Basic Activity.

点击Next，为应用程序命名（例如：My First App），选择Kotlin语言，然后点击Finish。Android Studio将使用系统中最新的API Level创建应用程序，并使用Gradle作为构建系统，在底部的视窗中可以查看整个过程。

### 2.探索Android Studio的界面布局
整个Android Studio工作区包括多个部分

### 3.创建模拟器
创建可以运行APP的模拟器，点击Tool>Device Manager或者工具栏上的按钮

点击Create device，弹出创建模拟器的页面，

创建可以运行APP的模拟器，点击Tool>Device Manager或者工具栏上的按钮在这里插入图片描述

点击Create device，弹出创建模拟器的页面，

然后首先下载镜像（Download），下载完成之后点击Next，完成模拟器命名和更多参数选择，最终点击Finish完成。注意：真实型号机型的模拟器镜像往往十分巨大，如果硬盘空间不足，考虑下载通用模拟器镜像。
### 4.在模拟器上运行应用程序
选择Run>Run ‘app’，在工具栏上可以看到运行程序的一些选择项。
### 5.查看布局编辑器
在Basic Activity中，包含了基本的导航组件，Android app关联两个fragments，第一个屏幕显示了“Hello first fragment”由FirstFragment创建，界面元素的排列由布局文件指定，查看res>layout>fragment_first.xml，

查看布局的代码（Code），修改Textview的Text属性

```
android:text="@string/hello_first_fragment"
```

右键该代码，选择Go To > Declaration or Usages，跳转到values/strings.xml，看到高亮文本
```
<string name="hello_first_fragment">Hello first fragment</string>
```

修改字符串属性值为“Hello Kotlin!”。更进一步，修改字体显示属性，在Design视图中选择textview_first文本组件，在Common Attributes属性下的textAppearance域，设置相关的文字显示属性，

查看布局的XML代码，可以看到新属性被应用。
```
android:fontFamily="sans-serif-condensed"
android:text="@string/hello_first_fragment"
android:textColor="@android:color/darker_gray"
android:textSize="30sp"
android:textStyle="bold"
```
重新运行应用程序，查看显示效果.

### 6.向页面添加更多的布局
本步骤将向第一个Fragment添加更多的视图组件

## (1)查看视图的布局约束
在fragment_first.xml，查看TextView组件的约束属性：
在约束布局中，每个组件至少需要一个水平方向和一个垂直方向的约束，更多约束布局的内容，请查看ConstraintLayout。

## (2)添加按钮和约束
从Palette面板中拖动Button到
调整Button的约束，设置Button的Top>BottonOf textView，
```
app:layout_constraintTop_toBottomOf="@+id/textview_first" />
```
随后添加Button的左侧约束至屏幕的左侧，Button的底部约束至屏幕的底部。查看Attributes面板，修改将id从button修改为toast_button（注意修改id将重构代码）

## （3）调整Next按钮
Next按钮是工程创建时默认的按钮，查看Next按钮的布局设计视图，它与TextView之间的连接不是锯齿状的而是波浪状的，表明两者之间存在链（chain），是一种两个组件之间的双向联系而不是单向联系。删除两者之间的链，可以在设计视图右键相应约束，选择Delete（注意两个组件要双向删除）；或者在属性面板的Constraint Widget中移动光标到相应约束点击删除。同时，删除Next按钮的左侧约束。

## （4）添加新的约束
添加Next的右边和底部约束至父类屏幕（如果不存在的话），Next的Top约束至TextView的底部。最后，TextView的底部约束至屏幕的底部。效果看起来如下图所示：

## （5）更改组件的文本
fragment_first.xml布局文件代码中，找到toast_button按钮的text属性部分
```
<Button
   android:id="@+id/toast_button"
   android:layout_width="wrap_content"
   android:layout_height="wrap_content"
   android:text="Button"

```
这里text的赋值是一种硬编码，点击文本，左侧出现灯泡状的提示，选择 Extract string resource。
弹出对话框，令资源名为toast_button_text，资源值为Toast，并点击OK。
于是，在资源文件string.xml定义了字符串，以上操作可以手动在string.xml文件中定义并引用。
```
<resources>
   ... 
   <string name="toast_button_text">Toast</string>
</resources>

```

## （4）更新Next按钮
在属性面板中更改Next按钮的id，从button_first改为random_button。
在string.xml文件，右键next字符串资源，选择 Refactor > Rename，修改资源名称为random_button_text，点击Refactor 。随后，修改Next值为Random。

## （5）添加第三个按钮
向fragment_first.xml文件中添加第三个按钮，位于Toast和Random按钮之间，TextView的下方。新Button的左右约束分别约束至Toast和Random，Top约束至TextView的底部，Buttom约束至屏幕的底部，看起来的效果：
检查xml代码，确保不出现类似app:layout_constraintVertical_bias这样的属性，即不手动设置偏移量。

## （6）完善UI组件的属性设置
更改新增按钮id为count_button，显示字符串为Count，对应字符串资源值为count_button_text。于是三个按钮的text和id属性如下表：
同时，更改TextView的文本为0。修改后的fragment_first.xml的代码
```
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".FirstFragment">

    <TextView
        android:id="@+id/textview_first"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:fontFamily="sans-serif-condensed"
        android:text="@string/hello_first_fragment"
        android:textColor="@android:color/darker_gray"
        android:textSize="30sp"
        android:textStyle="bold"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent" />

    <Button
        android:id="@+id/random_button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="@string/random_button_text"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintTop_toBottomOf="@+id/textview_first" />

    <Button
        android:id="@+id/toast_button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="@string/toast_button_text"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@+id/textview_first" />

    <Button
        android:id="@+id/count_button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="@string/count_button_text"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toStartOf="@+id/random_button"
        app:layout_constraintStart_toEndOf="@+id/toast_button"
        app:layout_constraintTop_toBottomOf="@+id/textview_first" />
</androidx.constraintlayout.widget.ConstraintLayout>

```
尝试运行应用程序查看效果。

### 7.更新按钮和文本框的外观

## （1）添加新的颜色资源
values>colors.xml定义了一些应用程序可以使用的颜色，添加新颜色screenBackground 值为 #2196F3，这是蓝色阴影色；添加新颜色buttonBackground 值为 #BBDEFB
```
<color name="screenBackground">#2196F3</color>
<color name="buttonBackground">#BBDEFB</color>

```

## （2）设置组件的外观
1.fragment_first.xml的属性面板中设置屏幕背景色为
```
android:background="@color/screenBackground"

```

2.设置每个按钮的背景色为buttonBackground
```
android:background="@color/buttonBackground"

```
注意：在实验的API level中（31），这种设置并不生效，需修改res/values/themes.xml的style值，添加**.Bridge**。
```
<style name="Theme.MyFirstApp" parent="Theme.MaterialComponents.DayNight.DarkActionBar.Bridge">

```
3.移除TextView的背景颜色，设置TextView的文本颜色为color/white，并增大字体大小至72sp

## （4）设置组件的位置
1.Toast与屏幕的左边距设置为24dp，Random与屏幕的右边距设置为24dp，利用属性面板的Constraint Widget完成设置。

2.设置TextView的垂直偏移为0.3，
```
app:layout_constraintVertical_bias="0.3"
```
拖动左侧的移动条。

### 运行应用程序
最终效果如下图：
- <img src="https://github.com/XIAOFA6/woyuyule/blob/main/img/a1.png" />

- <img src="https://github.com/XIAOFA6/woyuyule/blob/main/img/a2.png" />

- <img src="https://github.com/XIAOFA6/woyuyule/blob/main/img/a3.png" />

- <img src="https://github.com/XIAOFA6/woyuyule/blob/main/img/a4.png" />

- <img src="https://github.com/XIAOFA6/woyuyule/blob/main/img/a5.png" />

- <img src="https://github.com/XIAOFA6/woyuyule/blob/main/img/a6.png" />


