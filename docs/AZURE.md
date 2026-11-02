# Azure deployment

Deploy this repository as a **Linux Azure App Service web app**. Use the saved model; training does not run when the app starts.

1. In the [Azure portal](https://portal.azure.com), create **Web App**. Select your subscription, a new resource group, a unique app name, **Publish: Code**, **Python 3.12** and **Linux**. Choose an available region. A Basic B1 plan is sufficient for this small demonstration; review its displayed cost before creating it. No database, container registry or application secret is required.
2. Open the created App Service. In **Configuration / General settings**, set the startup command to:

   ```text
   gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 --no-control-socket --access-logfile - --error-logfile - app:app
   ```

3. In **Environment variables / App settings**, add `SCM_DO_BUILD_DURING_DEPLOYMENT` = `true`. Save the settings. Keep HTTPS enabled.
4. Open **Deployment Center**. Select **GitHub**, authorize your account, then choose owner `PansiluAW`, repository `colombo-rain-study-ML`, branch `main` and **GitHub Actions**. Select **User-assigned identity** if your subscription permissions allow Azure to create it. Preview the generated Python workflow, confirm Python **3.12**, then save. Azure adds the workflow and configures deployment authentication. No credentials should be pasted into source files.
5. Wait for the workflow in the repository's **Actions** tab to finish successfully. Open the App Service's **Default domain** from Overview. Use the actual domain shown by Azure, including any generated suffix.
6. Check the deployed app before recording:

   | Action | Expected result |
   | --- | --- |
   | Open `/health` | `{"status":"ok"}` |
   | Select Correct dry day, 2025-01-04 | DRY, 29.6% |
   | Change only humidity from 76 to 86 and submit | DRY, 62.0%; explanation updates; recorded outcome disappears |
   | Make minimum temperature 40, then submit | A temperature-consistency error; no prediction |
   | Open Test results and The data | All six figures load; CSV download works |

7. Open the deployed address in a private browser window and on a phone. Confirm examiners can access it without signing in. Put the real Azure URL alongside the GitHub URL in your Moodle submission and show it in the video. Keep the service available through marking.

If the default Azure welcome page appears, check the deployed branch, successful workflow and startup command. For a 500/503 error, open **Log stream**; confirm dependencies were installed and that `app.py`, `templates`, `static`, `src`, `models/qda.json`, `data/model_dataset.csv` and `results` were deployed. If identity setup is denied, an Azure subscription owner must grant the required role or configure the identity; changing the model code will not fix a subscription permission error.

After marking, remove the resource group when you no longer need the app to stop ongoing hosting charges.

Microsoft documentation: [Python App Service](https://learn.microsoft.com/en-us/azure/app-service/quickstart-python), [startup configuration](https://learn.microsoft.com/en-us/azure/app-service/configure-language-python), [GitHub continuous deployment](https://learn.microsoft.com/en-us/azure/app-service/deploy-continuous-deployment).
